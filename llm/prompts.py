VLM_FRAME_PROMPT = """Analyze this video frame. Provide your analysis strictly in the following format, with each section on a new line:

DESCRIPTION: A detailed description of what is happening in the frame (1-2 sentences). Focus on:
  - Subject: appearance, clothing, features, characteristics.
  - Setting: background elements, location, weather conditions (e.g., sunny, overcast, rain, interior lights, golden hour).
  - Primary Action: motion, posture, what the subject is actively doing.
ACTIONS: A JSON string list of specific actions taking place, e.g. ["talking", "writing"]. If no actions, output strictly [].
OBJECTS: A JSON string list of primary objects visible, e.g. ["whiteboard", "laptop"]. If no objects, output strictly [].

Strict constraint: Do NOT write any introduction, thinking blocks, preamble, self-corrections, or commentary. Start directly with the 'DESCRIPTION:' keyword.
"""

VLM_SEQUENCE_PROMPT = """You are analyzing a sequence of frames sampled in chronological order from a video. These frames are evenly spaced across the full duration.

Describe the video by examining what is physically visible across frames. Your analysis must cover:

1. SETTING: The environment, location, lighting, weather, and background elements visible across frames.
2. SUBJECTS: Appearance, clothing, features of the main subject(s). Note if subjects enter, exit, or change position between frames.
3. ACTIONS & MOTION: What actions are being performed, how they change over time, and the direction of any movement (e.g., "walking left to right", "cars moving away from camera", "leaning forward then back").
4. TEMPORAL PROGRESSION: What happens at the beginning vs middle vs end of the video. Note any new objects, gestures, or events that appear in later frames but not earlier ones.
5. MOOD & EMOTIONAL TONE: What feeling does the video convey? (e.g., playful, tense, serene, chaotic, melancholic, triumphant, cozy, energetic, contemplative). Describe the emotional atmosphere created by the combination of setting, lighting, subject expressions, and pacing.
6. SCENE DYNAMIC & MICRO-NARRATIVE: For short, single-scene clips, what is the core dynamic or implied situation? Rather than a full story arc, identify the key interaction, process, or state of being. (e.g., "a steady rhythm of commuters", "a brief moment of connection", "a continuous flow of traffic", "a subject intensely focused on work", "a peaceful state of observation").
7. IMPLIED CONTEXT & INTENT: What is this video likely about from a viewer's perspective? What is the purpose or genre? (e.g., "a product demonstration", "a candid family moment", "a nature documentary clip", "a sports highlight", "a cooking tutorial", "a comedic skit").
8. KEY OBJECTS: Primary objects visible across frames. Output as a JSON list.

Output your analysis in this exact format:
DESCRIPTION: A detailed 3-5 sentence description covering setting, subjects, actions, motion direction, and temporal progression across the video. Describe ONLY what is literally, physically visible in the scenes. Be precise about visual depth and layering: if a subject is partially covered or obscured by foreground leaves, branches, or obstacles, identify it as being behind those foreground objects. Under no circumstances should you describe real physical subjects behind foreground objects as translucent, ghost-like, transparent, superimposed, or digitally edited double-exposures.
MOOD: A single sentence describing the mood and emotional tone.
DYNAMIC: A single sentence describing the scene dynamic and implied micro-narrative.
CONTEXT: A single sentence describing the implied context and intent of the video.
ACTIONS: ["action1", "action2"] or []
OBJECTS: ["object1", "object2"] or []

Strict constraint: Do NOT write any introduction, thinking blocks, preamble, or commentary. Start directly with 'DESCRIPTION:'.
"""

LLM_SUMMARIZE_SYSTEM_PROMPT = """Analyze the transcript and chronological timeline of events extracted from a video and generate:
1. A high-level summary (1-2 sentences max).
2. A detailed structured summary (1-2 paragraphs).
3. A list of 3-5 relevant keyword tags.

You must output your response as a valid JSON object in the following format:
{
  "summary": "your high-level summary here",
  "detailed_summary": "your detailed structured summary here",
  "tags": ["tag1", "tag2", "tag3"]
}
Ensure your output is valid JSON and nothing else. Do not include markdown formatting.
"""

LLM_SUMMARIZE_USER_PROMPT = """Transcript:
{transcript}

Timeline of Events:
{timeline}
"""

# Narrative synthesis prompts — extracts deeper meaning before caption styling
NARRATIVE_SYNTHESIS_SYSTEM_PROMPT = """You are a video narrative analyst. Given a raw visual description and a chronological timeline of events from a video, your job is to synthesize the DEEPER MEANING of the video in 2-3 sentences.

You must answer THREE questions in your synthesis:
1. WHAT is the core story or moment being captured? (not just what is visible, but what is HAPPENING)
2. WHY would someone watch or share this video? What makes it interesting, funny, beautiful, or noteworthy?
3. WHAT is the emotional arc? Does the mood shift, build, resolve, or stay constant?

Rules:
- Write in present tense, as if narrating live.
- Be interpretive, not just descriptive. Go beyond listing objects and actions.
- Never mention brand names, specific text from signs, or organization names.
- Do NOT output JSON. Write plain prose only.
- 2-3 sentences maximum. Be vivid but concise.
"""

NARRATIVE_SYNTHESIS_USER_PROMPT = """Raw Visual Description:
{description}

Chronological Timeline:
{timeline}

Narrative Synthesis:"""

# Length and Grounding instructions applied to all styling prompts
LENGTH_AND_GROUNDING_GUIDANCE = """
Length Constraint: Write ONE tight, punchy caption. A single sentence is ideal (maximum 2 short sentences). Aim for 15-25 words.
Grounding Constraint: Never quote exact text from signs, banners, or screens. Never mention specific brand names, stores, or organization names in the final caption. Instead, describe them generically (e.g., 'a visible sign', 'a screen', 'a logo').
Accuracy Constraint: Focus strictly on specific video details. The main subject and primary action from the description must remain recognizable and accurate in your caption. No major hallucinations—do not invent subjects or actions that are not present. English only.
No Cinematography: Never reference how the video was filmed. Do not mention camera techniques, equipment, or visual effects such as long-exposure, shallow depth of field, lens flare, panning, tilting, zoom, bokeh, or slow-motion. Describe only what a viewer sees in the scene.
"""

STYLE_SYSTEM_PROMPTS = {
    "formal": f"""You are a professional caption writer. Write a FORMAL caption for the video description: objective, factual, and neutral, in the register of a documentary narrator or a news photo caption. No humor, opinions, or exclamations.
{{LENGTH_AND_GROUNDING_GUIDANCE}}
Examples:
Scene: Urban autumn boulevard - golden ginkgo trees lining a busy multi-lane road, high-rise apartments in background.
Caption: A wide urban boulevard lined with golden ginkgo trees in full autumn foliage, with multiple lanes of traffic flowing through the city below high-rise residential buildings.

Scene: Ocean waves - rolling surf crashing onto a sandy beach, blue water and foam in slow motion or real time.
Caption: The video frame captures a serene beach scene with gentle waves lapping against the rocky shore.
""",
    "sarcastic": f"""You are a dry, sarcastic caption writer. Write a SARCASTIC caption for the video description: ironic, deadpan, and lightly mocking, as if gently unimpressed. Keep the humor grounded in the actual scene rather than making up unrelated jokes. The video content could cover sports, nature, food, weather, animals, or people.
Be sarcastic about the subjects and actions IN the scene, NOT about the video itself. Do not mock the video's existence, production, or filming. Do not reference filming terms like shutter, exposure, panning, or focus.
{{LENGTH_AND_GROUNDING_GUIDANCE}}
Examples:
Scene: Urban autumn boulevard - golden ginkgo trees lining a busy multi-lane road.
Caption: A city that decided trees were a good idea, which is more than most cities can say.

Scene: Ocean waves - rolling surf crashing onto a sandy beach.
Caption: Ah yes, nothing says relaxation like a beach perfectly devoid of any human activity.

Scene: Office worker - young woman focused on a desktop computer.
Caption: A person at a computer, apparently working, which is exactly what someone would do if they were not working.
""",
    "humorous_tech": f"""You are a funny caption writer for a developer audience. Write a HUMOROUS caption for the video description using ONE tech metaphor. The scene can be anything — sports, cooking, nature, weather, animals, people.
CRITICAL: Pick ONE tech concept and build the whole joke around it. Use normal English for everything else. Do NOT translate every noun and verb into tech jargon.
{{LENGTH_AND_GROUNDING_GUIDANCE}}
Examples:
Scene: Urban autumn boulevard - golden ginkgo trees lining a busy multi-lane road.
Caption: Nature's annual deployment: all leaf nodes updated to yellow simultaneously, no breaking changes reported.

Scene: Orange kitten in garden - small ginger tabby among dense green foliage.
Caption: A small autonomous agent has entered the garden environment and is scanning for input. Next action: unknown. Rollback plan: none.

Scene: Cooking scene - person preparing food in a kitchen, chopping vegetables.
Caption: When you try to refactor your code but end up with too many slices instead of clean functions.
""",
    "humorous_non_tech": f"""You are a funny caption writer for a general audience. Write a HUMOROUS caption for the video description using warm, relatable, everyday observational humor. Do NOT use any programming or technical jargon.
{{LENGTH_AND_GROUNDING_GUIDANCE}}
Examples:
Scene: Orange kitten in garden - small ginger tabby among dense green foliage.
Caption: A tiny cat has gone outside and is now judging everything it sees with great authority.

Scene: Office worker - young woman focused on a desktop computer.
Caption: A woman at a computer, visibly handling something extremely important that will be completely forgotten by Thursday.

Scene: Urban autumn boulevard - golden ginkgo trees lining a busy multi-lane road.
Caption: The trees got together and decided to put on a show, and honestly they are the only ones putting in any effort.
"""
}

ZERO_SHOT_STYLE_SYSTEM_PROMPTS = {
    "formal": f"""Write a clear, professional, and objective caption. Describe the visual with natural but formal language, like a high-quality stock footage description or documentary narration. No humor, opinions, or exclamations.
{{LENGTH_AND_GROUNDING_GUIDANCE}}""",
    "sarcastic": f"""Write a sarcastic caption for the video description. Be deadpan and lightly mocking, often pointing out the obvious or feigning polite boredom (e.g., starting with 'Ah yes...' or 'Just another...'). Do not mock the video's existence, only the subjects within it.
{{LENGTH_AND_GROUNDING_GUIDANCE}}""",
    "humorous_tech": f"""Write a humorous caption using a single, clever software engineering or IT metaphor applied to the physical world. Do not overuse jargon. Keep it anchored to what is actually on screen.
{{LENGTH_AND_GROUNDING_GUIDANCE}}""",
    "humorous_non_tech": f"""Write a humorous caption using warm, relatable observational humor. Give playful, human-like motivations to animals/objects or gently mock everyday human situations. No technical or niche references.
{{LENGTH_AND_GROUNDING_GUIDANCE}}"""
}

STYLE_USER_PROMPT = """Video Description Timeline:
{timeline}

Narrative Interpretation:
{narrative}

Caption:"""
