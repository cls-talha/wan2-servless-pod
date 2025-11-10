import os
import io
import base64
import uuid
import logging
from datetime import timedelta
from PIL import Image
import torch
import runpod
from google.cloud import storage
import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import save_video
import random
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wan-i2v-serverless")

WAN_CHECKPOINT_DIR = "./Wan2.2-I2V-A14B"
LIGHTNING_DIR = "./Wan2.2-Lightning"
LORA_KEEP = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"
PIPELINE = None
PIPELINE_CFG = WAN_CONFIGS["i2v-A14B"]
DEVICE = 0
CKPT_DIR = WAN_CHECKPOINT_DIR
LORA_DIR = os.path.join(LIGHTNING_DIR, LORA_KEEP)
OFFLOAD_MODEL = True
BASE_SEED = random.randint(0, 999999)
SAVE_DIR = "test_results"
os.makedirs(SAVE_DIR, exist_ok=True)

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

def get_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE
    PIPELINE = wan.WanI2V(
        config=PIPELINE_CFG,
        checkpoint_dir=CKPT_DIR,
        lora_dir=LORA_DIR,
        device_id=DEVICE,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    return PIPELINE

def save_video_to_file(video, save_path, fps: float):
    save_video(tensor=video[None], save_file=save_path, fps=fps, nrow=1, normalize=True, value_range=(-1, 1))

def upload_to_gcs_public(source_file, bucket_name="runpod_bucket_testing"):
    gcs_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not gcs_json:
        raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS_JSON env variable")
    creds_path = "/tmp/gcs_creds.json"
    with open(creds_path, "w") as f:
        f.write(gcs_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    destination_blob = f"videos/{uuid.uuid4()}.mp4"
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_file)
    url = blob.generate_signed_url(expiration=timedelta(hours=1))
    return url

def generate_i2v(job):
    try:
        inputs = job.get("input", {})
        prompt = inputs.get("prompt", "No prompt")
        image_base64 = inputs.get("image_base64")
        frame_num = int(inputs.get("frame_num", 21))
        sampling_steps = int(inputs.get("sampling_steps", 6))

        if not image_base64:
            return {"status": "failed", "error": "Missing image_base64 input"}

        pipeline = get_pipeline()
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with torch.no_grad():
            video = pipeline.generate(
                prompt,
                img,
                max_area=MAX_AREA_CONFIGS["1280*720"],
                frame_num=frame_num,
                shift=5.0,
                sample_solver="euler",
                sampling_steps=sampling_steps,
                guide_scale=(1.0,1.0),
                seed=BASE_SEED,
                offload_model=OFFLOAD_MODEL,
            )

        filename = f"{uuid.uuid4()}.mp4"
        save_path = os.path.join(SAVE_DIR, filename)
        save_video_to_file(video, save_path, fps=PIPELINE_CFG.sample_fps)
        del video
        torch.cuda.synchronize()

        gcs_url = upload_to_gcs_public(save_path)
        os.remove(save_path)

        return {"status": "success", "gcs_url": gcs_url}

    except Exception as e:
        logger.exception("Generation failed")
        return {"status": "failed", "error": str(e)}

runpod.serverless.start({"handler": generate_i2v})
