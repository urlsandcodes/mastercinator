import streamlit as st
import asyncio
import logging
import json
import os
import time
from pathlib import Path
from app.manager import process_videos
from app.config import config
from app.logger import logger as app_logger

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="CineScribe AI — Cinematic Video Captioning",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern premium dark-mode styling and micro-animations
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F, #FFC7C7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        animation: fadeIn 1s ease-in-out;
    }
    .subtitle {
        color: #A0A5B5;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        border-bottom: 2px solid #33363F;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    /* Premium style caption card */
    .caption-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        border: 1px solid #2D3139;
        background: linear-gradient(145deg, #1B1E24, #121418);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    .caption-card:hover {
        transform: translateY(-3px);
        border-color: #FF4B4B;
        box-shadow: 0 8px 24px rgba(255, 75, 75, 0.15);
    }
    .style-header {
        font-size: 0.9rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    /* Style-specific header coloring */
    .style-formal { color: #5C7CFA; }
    .style-sarcastic { color: #FF922B; }
    .style-humorous_tech { color: #20C997; }
    .style-humorous_non_tech { color: #DA77F2; }
    .style-custom { color: #FCC419; }
    
    .caption-content {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #E2E8F0;
    }
    
    /* Performance metric badge */
    .metric-badge {
        background-color: #1A1D24;
        border: 1px solid #2D3139;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #20C997;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #A0A5B5;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Timeline styling */
    .timeline-item {
        border-left: 3px solid #FF4B4B;
        margin-left: 10px;
        padding-left: 20px;
        padding-bottom: 1.5rem;
        position: relative;
    }
    .timeline-item::before {
        content: '';
        width: 12px;
        height: 12px;
        background-color: #FF4B4B;
        border-radius: 50%;
        position: absolute;
        left: -8px;
        top: 4px;
        border: 2px solid #0E1117;
    }
    .timeline-time {
        font-weight: 700;
        color: #FF8F8F;
        font-size: 0.95rem;
    }
    .timeline-desc {
        color: #E2E8F0;
        margin-top: 0.25rem;
        font-size: 1rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# 2. Setup Session State
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "result" not in st.session_state:
    st.session_state["result"] = None
if "processing" not in st.session_state:
    st.session_state["processing"] = False

class StreamlitLogHandler(logging.Handler):
    """Intercepts framework logs and updates st.session_state["logs"] for real-time visualization."""
    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        log_entry = self.format(record)
        st.session_state["logs"].append(log_entry)

# 3. Setup VLM/LLM Provider Silently based on Environment API Key
if config.fireworks_api_key:
    config.vlm_provider = "fireworks"
else:
    config.vlm_provider = "mock"

# Sidebar for pipeline configuration options
with st.sidebar:
    st.header("Pipeline Configuration")
    pipeline_mode_ui = st.radio(
        "Select Pipeline Mode",
        options=["qwen_direct", "modular"],
        index=0 if config.pipeline_mode == "qwen_direct" else 1,
        help="qwen_direct runs a single-stage Qwen VL multimodal pipeline without audio transcription. modular runs the visual VLM timeline + narrative synthesis + styling pipeline."
    )
    config.pipeline_mode = pipeline_mode_ui

# 4. Main Panel Layout
st.markdown('<div class="main-title">CineScribe</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CineScribe is an advanced video intelligence agent that combines multi-modal visual timeline analysis and audio transcripts to generate highly accurate, style-conditioned video captions across varied contexts.</div>', unsafe_allow_html=True)

# Main panel video uploader (accepts only .mp4)
uploaded_file = st.file_uploader("Upload Video File (MP4, Max 20MB)", type=["mp4"])

run_clicked = False
final_video_source = None

if uploaded_file is not None:
    if uploaded_file.size > 20 * 1024 * 1024:
        st.error("File size exceeds the 20MB limit. Please upload a smaller video.")
    else:
        # Show button only after upload succeeds
        run_clicked = st.button("🎬 Generate Styled Captions", type="primary", disabled=st.session_state["processing"])

if run_clicked and uploaded_file is not None:
    # Save the file locally
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / uploaded_file.name
    with st.spinner("Saving uploaded file locally..."):
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    final_video_source = str(temp_file_path.resolve())
    
    # Active styles are fixed (no selector needed)
    active_styles = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
    
    st.session_state["processing"] = True
    st.session_state["result"] = None
    st.session_state["logs"] = []
    
    # Async runner wrapped inside streamlit
    async def main_runner():
        task = {
            "task_id": "demo-session",
            "video_url": final_video_source,
            "styles": active_styles
        }
        # Run the backend execution manager
        results = await process_videos([], max_parallel=1, tasks=[task])
        return results[0]

    try:
        with st.spinner("Processing video... Please wait..."):
            result = asyncio.run(main_runner())
            
        if result.status == "failed":
            st.error("Pipeline execution failed. Please check the logs in terminal/docker stdout.")
            if result.stage_errors:
                st.json(result.stage_errors)
        else:
            st.success("Successfully processed video and aligned styles!")
            st.session_state["result"] = result
    except Exception as ex:
        st.exception(ex)
    finally:
        st.session_state["processing"] = False

# 5. Render Results Dashboard
if st.session_state["result"]:
    res = st.session_state["result"]
    
    st.markdown('<div class="section-header">🎬 Generated Style-Conditioned Captions</div>', unsafe_allow_html=True)
    
    # Iterate over and show the styled captions
    if res.captions:
        for style, text in res.captions.items():
            nice_name = style.replace("_", " ").title()
            header_class = f"style-{style}" if style in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"] else "style-custom"
            
            icon = "👔" if style == "formal" else "😏" if style == "sarcastic" else "💻" if style == "humorous_tech" else "😄" if style == "humorous_non_tech" else "✨"
            
            st.markdown(f"""
            <div class="caption-card">
                <div class="style-header {header_class}">{icon} {nice_name}</div>
                <div class="caption-content">{text}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No captions generated.")
        
    # Add a download results button
    result_json_str = json.dumps({
        "task_id": res.id,
        "captions": res.captions or {}
    }, indent=2)
    st.download_button(
        label="💾 Download Captions JSON",
        data=result_json_str,
        file_name=f"cinescribe_{res.id}_results.json",
        mime="application/json",
        use_container_width=True
    )
