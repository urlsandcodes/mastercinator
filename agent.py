import os
import sys
import json
import tempfile
import base64
import re
import requests
import cv2
from openai import OpenAI
from dotenv import load_dotenv

# Load environmental configurations
load_dotenv(override=True)

# Global Client setup pointing to Fireworks AI
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ.get("FIREWORKS_API_KEY")
)

# Shared Grounding Constraints
LENGTH_AND_GROUNDING_GUIDANCE = """
Length Constraint: Write ONE tight, punchy caption. A single sentence is ideal (maximum 2 short sentences). Aim for 15-25 words.
Grounding Constraint: Never quote exact text from signs, banners, or screens. Never mention specific brand names, stores, or organization names in the final caption. Instead, describe them generically (e.g., 'a visible sign', 'a screen', 'a logo').
Accuracy Constraint: Focus strictly on specific video details. The main subject and primary action from the description must remain recognizable and accurate in your caption. No major hallucinations—do not invent subjects or actions that are not present. English only.
No Cinematography: Never reference how the video was filmed. Do not mention camera techniques, equipment, or visual effects such as long-exposure, shallow depth of field, lens flare, panning, tilting, zoom, bokeh, or slow-motion. Describe only what a viewer sees in the scene.
"""

GENERATE_SYSTEM_PROMPT = """You are a strict data-formatting pipeline. You will receive a set of video frames.
You MUST output a valid JSON object containing exactly these keys: "formal", "sarcastic", "humorous_tech", "humorous_non_tech".
Do NOT output any reasoning, markdown blocks, thinking blocks, or conversational text. Output ONLY raw JSON."""

GENERATE_USER_PROMPT = f"""Write a caption for the video shown in the keyframes for each of these 4 styles. 

Style Guidelines & Few-Shot Examples:

1. **formal**: Objective, factual, and neutral, in the register of a documentary narrator. No humor, opinions, or exclamations.
{LENGTH_AND_GROUNDING_GUIDANCE}
Examples:
- Scene: Urban autumn boulevard - ginkgo trees lining a multi-lane road, high-rise apartments in background.
  Caption: A wide urban boulevard lined with golden ginkgo trees in full autumn foliage, with multiple lanes of traffic flowing through the city below high-rise residential buildings.
- Scene: Ocean waves - surf crashing onto a sandy beach, blue water and foam.
  Caption: The video captures a serene beach scene with gentle waves lapping against the rocky shore.

2. **sarcastic**: Dry, ironic, deadpan, and lightly mocking, as if gently unimpressed. Keep the humor grounded in the actual scene.
{LENGTH_AND_GROUNDING_GUIDANCE}
Examples:
- Scene: Urban autumn boulevard - golden ginkgo trees lining a busy multi-lane road.
  Caption: A city that decided trees were a good idea, which is more than most cities can say.
- Scene: Ocean waves - rolling surf crashing onto a sandy beach.
  Caption: Ah yes, nothing says relaxation like a beach perfectly devoid of any human activity.
- Scene: Office worker - young woman focused on a desktop computer.
  Caption: A person at a computer, apparently working, which is exactly what someone would do if they were not working.

3. **humorous_tech**: A funny caption for a developer audience using ONE tech metaphor. Build the whole joke around it.
{LENGTH_AND_GROUNDING_GUIDANCE}
Examples:
- Scene: Urban autumn boulevard - golden ginkgo trees lining a busy multi-lane road.
  Caption: Nature's annual deployment: all leaf nodes updated to yellow simultaneously, no breaking changes reported.
- Scene: Orange kitten in garden - small ginger tabby among dense green foliage.
  Caption: A small autonomous agent has entered the garden environment and is scanning for input. Next action: unknown. Rollback plan: none.
- Scene: Cooking scene - person preparing food in a kitchen, chopping vegetables.
  Caption: When you try to refactor your code but end up with too many slices instead of clean functions.

4. **humorous_non_tech**: Warm, relatable, everyday observational humor. Do NOT use any programming or technical jargon.
{LENGTH_AND_GROUNDING_GUIDANCE}
Examples:
- Scene: Orange kitten in garden - small ginger tabby among dense green foliage.
  Caption: A tiny cat has gone outside and is now judging everything it sees with great authority.
- Scene: Office worker - young woman focused on a desktop computer.
  Caption: A woman at a computer, visibly handling something extremely important that will be completely forgotten by Thursday.
- Scene: Urban autumn boulevard - golden ginkgo trees lining a busy multi-lane road.
  Caption: The trees got together and decided to put on a show, and honestly they are the only ones putting in any effort.

Generate the 4 captions now in the requested JSON format."""

VERIFY_SYSTEM_PROMPT = """You are a video validation and correction pipeline. You will receive a set of video frames and a set of draft captions for those frames.
Verify if the subjects, actions, scenery, and objects described in the draft captions match the visible elements in the frames.
Identify any visual hallucinations, generic claims, brand names, landmarks, location claims, camera movement terms (exposure, zoom, panning), or digital effects.
For any incorrect claim, rewrite the caption to correct it while preserving the requested style (formal, sarcastic, humorous_tech, humorous_non_tech).
You MUST output a valid JSON object containing exactly the same keys: "formal", "sarcastic", "humorous_tech", "humorous_non_tech".
Do NOT output any markdown blocks, reasoning, or conversational text. Output ONLY raw JSON."""

VERIFY_USER_PROMPT = """Draft Captions:
{draft_captions_json}

Review the verification frames in detail and output the final validated and corrected JSON object:"""


def download_video(url, save_path):
    print(f"Downloading video from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download video: {e}")
        return False


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
    print("Extracting exactly 8 representative frames...")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video file.")
        return []
        
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("WARNING: Could not retrieve total frame count. Sequential reading instead.")
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

    # Standard temporal sampling
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
        else:
            print(f"Warning: Failed to read frame at index {frame_idx}")
            
    video.release()
    print(f"Extracted {len(selected_frames)} frames.")
    return selected_frames


def generate_draft_captions(gen_frames):
    print("Call 1: Generating draft captions using Qwen 3.7 Plus...")
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
        model="accounts/fireworks/models/qwen3p7-plus",
        messages=[
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        max_tokens=1000,
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content.strip()


def verify_and_critique_captions(verify_frames, draft_json_str):
    print("Call 2: Verifying and critiquing captions using Qwen 3.7 Plus...")
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
        model="accounts/fireworks/models/qwen3p7-plus",
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        max_tokens=1000,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content.strip()


def main():
    input_path = os.environ.get("TASKS_PATH", "/input/tasks.json")
    output_path = os.environ.get("RESULTS_PATH", "/output/results.json")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)
        
    with open(input_path, 'r', encoding="utf-8") as f:
        try:
            tasks = json.load(f)
        except Exception as e:
            print(f"Error: Malformed tasks.json file: {e}")
            sys.exit(1)
            
    print(f"Loaded {len(tasks)} tasks.")
    results = []
    
    for task in tasks:
        task_id = task.get("task_id")
        video_url = task.get("video_url")
        styles = task.get("styles", ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"])
        
        print(f"\n=== Processing Task: {task_id} ===")
        temp_video = tempfile.mktemp(suffix=".mp4")
        
        final_captions = {s: "" for s in styles}
        
        if download_video(video_url, temp_video):
            frames = extract_exact_8_frames(temp_video)
            
            if len(frames) == 8:
                # Divide into alternate frames
                gen_frames = [frames[0], frames[2], frames[4], frames[6]]
                verify_frames = [frames[1], frames[3], frames[5], frames[7]]
                
                try:
                    draft_json = generate_draft_captions(gen_frames)
                    print(f"Draft JSON: {draft_json}")
                    
                    verified_json = verify_and_critique_captions(verify_frames, draft_json)
                    print(f"Verified JSON: {verified_json}")
                    
                    parsed = json.loads(verified_json)
                    
                    # Fill final captions
                    for style in styles:
                        final_captions[style] = parsed.get(style, f"[CAPTION_FAILED: Missing {style} in response]")
                except Exception as e:
                    print(f"VLM pipeline failed for task {task_id}: {e}")
                    for style in styles:
                        final_captions[style] = f"[CAPTION_FAILED: VLM execution failed: {e}]"
            else:
                print(f"Failed to extract exactly 8 frames for task {task_id}")
                for style in styles:
                    final_captions[style] = f"[CAPTION_FAILED: Frame extraction failed]"
                    
            if os.path.exists(temp_video):
                os.remove(temp_video)
        else:
            print(f"Failed to download video for task {task_id}")
            for style in styles:
                final_captions[style] = f"[CAPTION_FAILED: Video download failed]"
                
        results.append({
            "task_id": task_id,
            "captions": final_captions
        })
        
    print("\nWriting final results to results.json...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("Execution complete.")


if __name__ == "__main__":
    main()
