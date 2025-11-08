import os
import io
import uuid
import base64
import logging
import gc
from datetime import datetime
from PIL import Image
import torch
import runpod
import subprocess

import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import save_video

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wan-i2v-serverless")

# --------------------
# Model / Lightning Setup
# --------------------
WAN_CHECKPOINT_DIR = "./Wan2.2-I2V-A14B"
LIGHTNING_DIR = "./Wan2.2-Lightning"
LORA_KEEP = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"

def setup_models():
    # WAN checkpoint
    if not os.path.exists(WAN_CHECKPOINT_DIR):
        logger.info("[SETUP] Downloading WAN I2V checkpoint...")
        subprocess.run([
            "huggingface-cli", "download",
            "Wan-AI/Wan2.2-I2V-A14B",
            "--local-dir", WAN_CHECKPOINT_DIR
        ], check=True)
    else:
        logger.info("[SETUP] WAN I2V checkpoint already exists.")

    # Lightning repo
    if not os.path.exists(LIGHTNING_DIR):
        logger.info("[SETUP] Downloading Wan2.2-Lightning repo...")
        subprocess.run([
            "huggingface-cli", "download",
            "lightx2v/Wan2.2-Lightning",
            "--local-dir", LIGHTNING_DIR
        ], check=True)
    else:
        logger.info("[SETUP] Wan2.2-Lightning already exists.")

    # Clean up Lightning folder: keep only LoRA folder
    for folder in os.listdir(LIGHTNING_DIR):
        folder_path = os.path.join(LIGHTNING_DIR, folder)
        if os.path.isdir(folder_path) and folder != LORA_KEEP:
            logger.info(f"[SETUP] Removing folder {folder_path}")
            subprocess.run(["rm", "-rf", folder_path], check=True)

# Run setup once
setup_models()


# -------------------- Global Config --------------------
PIPELINE = None
PIPELINE_CFG = WAN_CONFIGS["i2v-A14B"]
DEVICE = 0
RANK = 0
CKPT_DIR = "./Wan2.2-I2V-A14B"
LORA_DIR = "./Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"
OFFLOAD_MODEL = True
BASE_SEED = 42
SAVE_DIR = "test_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- Lazy Pipeline Loader --------------------
def get_pipeline():
    """Load the pipeline once and reuse it (GPU optimized)."""
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    logger.info("[LOAD] Initializing Wan I2V pipeline...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    PIPELINE = wan.WanI2V(
        config=PIPELINE_CFG,
        checkpoint_dir=CKPT_DIR,
        lora_dir=LORA_DIR,
        device_id=DEVICE,
        rank=RANK,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=True,
    )

    logger.info("[READY] WAN I2V model loaded successfully.")
    return PIPELINE

# -------------------- Helpers --------------------
def _format_filename(prompt: str, job_id: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = (prompt or "no_prompt").replace(" ", "_").replace("/", "_")[:50]
    return f"i2v_{job_id}_{safe_prompt}_{ts}.mp4"

def save_video_to_file(video, save_path, fps: float):
    save_video(
        tensor=video[None],
        save_file=save_path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )

# -------------------- RunPod Handler --------------------
def generate_i2v(job):
    """
    RunPod Serverless job handler
    Expected input:
    {
        "input": {
            "prompt": "...",
            "image_base64": "<base64 string>",
            "frame_num": 21,
            "sampling_steps": 6
        }
    }
    """
    try:
        inputs = job["input"]
        prompt = inputs.get("prompt")
        image_base64 = inputs.get("image_base64")
        frame_num = int(inputs.get("frame_num", 21))
        sampling_steps = int(inputs.get("sampling_steps", 6))

        # Hardcoded parameters (same as your API defaults)
        size = "1280*720"
        guide_scale = (1.0, 1.0)
        shift = 5.0
        sample_solver = "euler"

        if not image_base64:
            return {"error": "Missing image_base64 input"}

        if size not in SUPPORTED_SIZES["i2v-A14B"]:
            return {"error": f"Unsupported size {size}"}

        logger.info(f"[JOB {job['id']}] Loading pipeline and decoding image...")
        pipeline = get_pipeline()

        # Decode base64 → PIL Image
        try:
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"error": f"Invalid base64 image: {e}"}

        logger.info(f"[JOB {job['id']}] Generating video | frame_num={frame_num}, steps={sampling_steps}")

        with torch.no_grad():
            video = pipeline.generate(
                prompt,
                img,
                max_area=MAX_AREA_CONFIGS[size],
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=BASE_SEED,
                offload_model=OFFLOAD_MODEL,
            )

            filename = _format_filename(prompt, job["id"])
            save_path = os.path.join(SAVE_DIR, filename)
            save_video_to_file(video, save_path, fps=PIPELINE_CFG.sample_fps)

            del video
            torch.cuda.synchronize()

        logger.info(f"[JOB {job['id']}] Completed. Saved to {save_path}")
        return {"status": "success", "video_path": save_path}

    except Exception as e:
        logger.exception("Generation failed")
        return {"status": "failed", "error": str(e)}

# -------------------- RunPod Start --------------------
runpod.serverless.start({"handler": generate_i2v})
