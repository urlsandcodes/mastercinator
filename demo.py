import os
import tempfile
import base64
import json
import cv2
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv(override=True)

# Streamlit Page Setup - Minimal UI
st.set_page_config(page_title="MasterCinator", page_icon="🎬", layout="centered")
st.title("MasterCinator")

# Fireworks OpenAI client
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ.get("FIREWORKS_API_KEY")
)

def extract_json_block(text):
    text_clean = text.strip()
    if text_clean.startswith("```"):
        lines = text_clean.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text_clean = "\n".join(lines).strip()
    
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text_clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse valid JSON from text: {text}")

# Shared Grounding Constraints (Few-Shot)
LENGTH_AND_GROUNDING_GUIDANCE_FEW_SHOT = """
Length Constraint: Write ONE tight, punchy caption. A single sentence is ideal (maximum 2 short sentences). Aim for 15-25 words.
Grounding Constraint: Never quote exact text from signs, banners, or screens. Never mention specific brand names, stores, or organization names in the final caption. Instead, describe them generically (e.g., 'a visible sign', 'a screen', 'a logo').
Accuracy Constraint: Focus strictly on specific video details. The main subject and primary action from the description must remain recognizable and accurate in your caption. No major hallucinations—do not invent subjects or actions that are not present. English only.
No Cinematography: Never reference how the video was filmed. Do not mention camera techniques, equipment, or visual effects such as long-exposure, shallow depth of field, lens flare, panning, tilting, zoom, bokeh, or slow-motion. Describe only what a viewer sees in the scene.
"""

GENERATE_SYSTEM_PROMPT_FEW_SHOT = """You are a video captioning tool. Write 4 styled captions.
Your output MUST be a JSON object containing keys: "formal", "sarcastic", "humorous_tech", "humorous_non_tech".
Do NOT write any preambles, explanations, or step-by-step thinking in the text. Output ONLY the JSON block (inside ```json and ``` codeblock)."""

GENERATE_USER_PROMPT_FEW_SHOT = f"""Write a caption for the video shown in the keyframes for each of these 4 styles. 

Style Guidelines & Few-Shot Examples:

1. **formal**: Objective, factual, and neutral, in the register of a documentary narrator. No humor, opinions, or exclamations.
{LENGTH_AND_GROUNDING_GUIDANCE_FEW_SHOT}
Examples:
- Scene: Urban autumn boulevard - ginkgo trees lining a multi-lane road, high-rise apartments in background.
  Caption: A wide urban boulevard lined with golden ginkgo trees in full autumn foliage, with multiple lanes of traffic flowing through the city below high-rise residential buildings.
- Scene: Ocean waves - surf crashing onto a sandy beach, blue water and foam.
  Caption: The video captures a serene beach scene with gentle waves lapping against the rocky shore.

2. **sarcastic**: Dry, ironic, deadpan, and lightly mocking, as if gently unimpressed. Keep the humor grounded in the actual scene.
{LENGTH_AND_GROUNDING_GUIDANCE_FEW_SHOT}
Examples:
- Scene: Urban autumn boulevard - golden ginkgo trees lining a busy multi-lane road.
  Caption: A city that decided trees were a good idea, which is more than most cities can say.
- Scene: Ocean waves - rolling surf crashing onto a sandy beach.
  Caption: Ah yes, nothing says relaxation like a beach perfectly devoid of any human activity.
- Scene: Office worker - young woman focused on a desktop computer.
  Caption: A person at a computer, apparently working, which is exactly what someone would do if they were not working.

3. **humorous_tech**: A funny caption for a developer audience using ONE tech metaphor. Build the whole joke around it.
{LENGTH_AND_GROUNDING_GUIDANCE_FEW_SHOT}
Examples:
- Scene: Urban autumn boulevard - golden ginkgo trees lining a busy multi-lane road.
  Caption: Nature's annual deployment: all leaf nodes updated to yellow simultaneously, no breaking changes reported.
- Scene: Orange kitten in garden - small ginger tabby among dense green foliage.
  Caption: A small autonomous agent has entered the garden environment and is scanning for input. Next action: unknown. Rollback plan: none.
- Scene: Cooking scene - person preparing food in a kitchen, chopping vegetables.
  Caption: When you try to refactor your code but end up with too many slices instead of clean functions.

4. **humorous_non_tech**: Warm, relatable, everyday observational humor. Do NOT use any programming or technical jargon.
{LENGTH_AND_GROUNDING_GUIDANCE_FEW_SHOT}
Examples:
- Scene: Orange kitten in garden - small ginger tabby among dense green foliage.
  Caption: A tiny cat has gone outside and is now judging everything it sees with great authority.
- Scene: Office worker - young woman focused on a desktop computer.
  Caption: A woman at a computer, visibly handling something extremely important that will be completely forgotten by Thursday.
- Scene: Urban autumn boulevard - golden ginkgo trees lining a busy multi-lane road.
  Caption: The trees got together and decided to put on a show, and honestly they are the only ones putting in any effort.

Generate the 4 captions now in the requested JSON format."""

VERIFY_SYSTEM_PROMPT_FEW_SHOT = """You are a JSON correction filter. You will receive draft captions and a set of verification frames.
Your task is to correct any visual hallucinations, brand names, or location claims in the draft captions based on the frames.
You MUST output ONLY a valid JSON object with the keys "formal", "sarcastic", "humorous_tech", "humorous_non_tech".
Do NOT explain your corrections. Do NOT write down any step-by-step verification or reasoning in the text output. Output ONLY the JSON block, optionally enclosed in a ```json ``` block."""

VERIFY_USER_PROMPT_FEW_SHOT = """Draft Captions:
{draft_captions_json}

Review the verification frames in detail and output the final validated and corrected JSON object:"""

# Shared Grounding Constraints (Zero-Shot)
LENGTH_AND_GROUNDING_GUIDANCE = """
Length Constraint: Write ONE tight, punchy caption. A single sentence is ideal (maximum 2 short sentences). Aim for 15-25 words.
Grounding Constraint: Never quote exact text from signs, banners, or screens. Never mention specific brand names, stores, or organization names in the final caption. Instead, describe them generically (e.g., 'a visible sign', 'a screen', 'a logo').
Accuracy Constraint: Focus strictly on specific video details. The main subject and primary action from the description must remain recognizable and accurate in your caption. No major hallucinations—do not invent subjects or actions that are not present. English only.
No Cinematography: Never reference how the video was filmed. Do not mention camera techniques, equipment, or visual effects such as long-exposure, shallow depth of field, lens flare, panning, tilting, zoom, bokeh, or slow-motion. Describe only what a viewer sees in the scene.
Factual Grounding: The caption must include description of the video, and only what is happening in the video is to be referenced for generating styled captions.
"""

GENERATE_SYSTEM_PROMPT = """You are a video captioning pipeline. You will receive video frames.
Output a valid JSON object with exactly these keys: "formal", "sarcastic", "humorous_tech", "humorous_non_tech".
Do NOT output any reasoning, markdown blocks, or conversational text. Output ONLY raw JSON."""

GENERATE_USER_PROMPT = f"""Write a caption for the video shown in the keyframes for each of these 4 styles. 

Style Guidelines:

1. **formal**: Objective, factual, and neutral, in the register of a documentary narrator. No humor, opinions, or exclamations.
{LENGTH_AND_GROUNDING_GUIDANCE}

2. **sarcastic**: Dry, ironic, deadpan, and lightly mocking, as if gently unimpressed. Keep the humor grounded in the actual scene but you may use ironic comparisons or references to make your point.
{LENGTH_AND_GROUNDING_GUIDANCE}

3. **humorous_tech**: A funny caption for a developer audience using ONE tech metaphor. Build the whole joke around that single metaphor. The metaphor must map clearly to something visible in the video.
{LENGTH_AND_GROUNDING_GUIDANCE}

4. **humorous_non_tech**: Warm, relatable, everyday observational humor. Do NOT use any programming or technical jargon. You may use similes, comparisons, or everyday references to amplify the humor, but the observation must be rooted in the scene.
{LENGTH_AND_GROUNDING_GUIDANCE}

Generate the 4 captions now in the requested JSON format."""

VERIFY_SYSTEM_PROMPT = """You are a video validation and correction pipeline. You will receive video frames and draft captions.
Verify that the subjects, actions, scenery, and objects in the draft captions match the visible elements in the frames.
Identify any visual hallucinations, brand names, location claims, or cinematography terms. Correct them while preserving each caption's style.
Output a valid JSON object with exactly the keys: "formal", "sarcastic", "humorous_tech", "humorous_non_tech".
Do NOT output any markdown blocks, reasoning, or conversational text. Output ONLY raw JSON."""

VERIFY_USER_PROMPT = """Draft Captions:
{draft_captions_json}

Review the verification frames in detail and output the final validated and corrected JSON object:"""


def resize_frame(frame, target_longest_side=1024):
    height, width = frame.shape[:2]
    if max(height, width) <= target_longest_side:
        return frame
    
    if width > height:
        scale = target_longest_side / width
        new_width = target_longest_side
        new_height = int(height * scale)
    else:
        scale = target_longest_side / height
        new_height = target_longest_side
        new_width = int(width * scale)
        
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def extract_exact_8_frames(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return []
        
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback sequential read
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        video.release()
        
        if not frames:
            return []
            
        total_frames = len(frames)
        interval = max(1, total_frames // 8)
        selected_frames = []
        for i in range(8):
            idx = min(i * interval, total_frames - 1)
            resized = resize_frame(frames[idx], target_longest_side=1024)
            _, buffer = cv2.imencode('.jpg', resized)
            selected_frames.append(base64.b64encode(buffer).decode('utf-8'))
        return selected_frames

    interval = max(1, total_frames // 8)
    selected_frames = []
    for i in range(8):
        frame_idx = min(i * interval, total_frames - 1)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if ret:
            resized = resize_frame(frame, target_longest_side=1024)
            _, buffer = cv2.imencode('.jpg', resized)
            selected_frames.append(base64.b64encode(buffer).decode('utf-8'))
            
    video.release()
    return selected_frames


def generate_draft_captions(gen_frames):
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img}"
            }
        }
        for img in gen_frames
    ]
    content.append({
        "type": "text",
        "text": GENERATE_USER_PROMPT
    })
    
    response = client.chat.completions.create(
        model="accounts/fireworks/models/minimax-m3",
        messages=[
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        max_tokens=2000,
        temperature=0.7
    )
    
    raw_content = response.choices[0].message.content.strip()
    parsed = extract_json_block(raw_content)
    return json.dumps(parsed)


def verify_and_critique_captions(verify_frames, draft_json_str):
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img}"
            }
        }
        for img in verify_frames
    ]
    content.append({
        "type": "text",
        "text": VERIFY_USER_PROMPT.format(draft_captions_json=draft_json_str)
    })
    
    response = client.chat.completions.create(
        model="accounts/fireworks/models/minimax-m3",
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        max_tokens=2000,
        temperature=0.2
    )
    
    raw_content = response.choices[0].message.content.strip()
    parsed = extract_json_block(raw_content)
    return json.dumps(parsed)


# File uploader
video_file = st.file_uploader("Upload a video (max 10MB)", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Size check
    if video_file.size > 10 * 1024 * 1024:
        st.error("Video file is too large! Please upload a file smaller than 10MB.")
    else:
        st.success("Upload complete!")
        
        # Analyze Button
        if st.button("Analyze", type="primary"):
            with st.spinner("Processing video..."):
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                    temp_video.write(video_file.read())
                    temp_video_path = temp_video.name
                
                try:
                    # Extract frames
                    frames = extract_exact_8_frames(temp_video_path)
                    
                    if len(frames) == 8:
                        gen_frames = [frames[0], frames[2], frames[4], frames[6]]
                        verify_frames = [frames[1], frames[3], frames[5], frames[7]]
                        
                        # VLM Steps
                        draft_json = generate_draft_captions(gen_frames)
                        verified_json = verify_and_critique_captions(verify_frames, draft_json)
                        
                        # Output raw JSON block only
                        st.json(json.loads(verified_json))
                    else:
                        st.error("Error: Could not extract exactly 8 frames from the uploaded video.")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                finally:
                    # Cleanup
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
