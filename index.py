import sys
import os
import io
import gc

# Local patched BasicSR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "basicsr-patched"))

# CPU memory optimization
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import cv2
import requests
import numpy as np

from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ---------------------------------------------------------
# BasicSR compatibility
# ---------------------------------------------------------

from torchvision.transforms.functional import rgb_to_grayscale

import torchvision.transforms.functional as F

if not hasattr(F, "rgb_to_grayscale"):
    F.rgb_to_grayscale = rgb_to_grayscale


# ---------------------------------------------------------
# Real-ESRGAN
# ---------------------------------------------------------

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# ---------------------------------------------------------
# Flask
# ---------------------------------------------------------

app = Flask(__name__)
CORS(app)

MAX_FILE_SIZE = 10 * 1024 * 1024


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "weights",
    "RealESRGAN_x4plus.pth"
)

print("Loading Real-ESRGAN model...")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)


upsampler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,

    # IMPORTANT FOR LOW RAM
    tile=128,
    tile_pad=10,
    pre_pad=0,

    # CPU
    half=False
)


print("Real-ESRGAN model loaded successfully.")


# ---------------------------------------------------------
# Image processing
# ---------------------------------------------------------

def upscale_image(image_bytes):

    if not image_bytes:
        raise ValueError("Empty image")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            "Image size must be less than 10 MB"
        )

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Limit huge images
    max_dimension = 2048

    if max(image.size) > max_dimension:

        ratio = max_dimension / max(image.size)

        new_size = (
            max(1, int(image.width * ratio)),
            max(1, int(image.height * ratio))
        )

        image = image.resize(
            new_size,
            Image.Resampling.LANCZOS
        )

    image_np = np.asarray(
        image,
        dtype=np.uint8
    )

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    # Release PIL/numpy temporary memory
    del image
    del image_np

    gc.collect()

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    with torch.inference_mode():

        output, _ = upsampler.enhance(
            image_bgr,
            outscale=4
        )

    del image_bgr

    gc.collect()

    # BGR -> RGB
    output_rgb = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )

    del output

    gc.collect()

    result = Image.fromarray(output_rgb)

    del output_rgb

    buffer = io.BytesIO()

    result.save(
        buffer,
        format="PNG",
        optimize=True
    )

    result.close()

    buffer.seek(0)

    gc.collect()

    return buffer


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "service": "Rakib Real-ESRGAN API",
        "status": "online",
        "model": "RealESRGAN_x4plus",
        "scale": "4x",
        "device": "CPU",
        "memory_mode": "tile-128",
        "endpoints": {
            "health": "/health",
            "upscale": "POST /upscale",
            "upscale_url": "POST /upscale-url"
        }
    })


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "service": "Rakib Real-ESRGAN API",
        "model": "RealESRGAN_x4plus",
        "device": "CPU"
    })


# ---------------------------------------------------------
# Upload
# ---------------------------------------------------------

@app.route("/upscale", methods=["POST"])
def upscale():

    try:

        if "image" not in request.files:

            return jsonify({
                "status": "error",
                "message": "No image uploaded. Use field name: image"
            }), 400

        file = request.files["image"]

        if not file.filename:

            return jsonify({
                "status": "error",
                "message": "Invalid filename"
            }), 400

        image_bytes = file.read()

        result = upscale_image(image_bytes)

        return send_file(
            result,
            mimetype="image/png",
            as_attachment=False,
            download_name="upscaled.png"
        )

    except Exception as e:

        print(
            "Upscale error:",
            repr(e),
            flush=True
        )

        gc.collect()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------------------------------------------------
# URL
# ---------------------------------------------------------

@app.route("/upscale-url", methods=["POST"])
def upscale_url():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        url = data.get("url")

        if not url:

            return jsonify({
                "status": "error",
                "message": "Missing image URL"
            }), 400

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        if len(response.content) > MAX_FILE_SIZE:

            return jsonify({
                "status": "error",
                "message": "Remote image is too large"
            }), 400

        result = upscale_image(
            response.content
        )

        return send_file(
            result,
            mimetype="image/png",
            as_attachment=False,
            download_name="upscaled.png"
        )

    except Exception as e:

        print(
            "URL upscale error:",
            repr(e),
            flush=True
        )

        gc.collect()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------------------------------------------------
# Render
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", "5000")
    )

    print(
        f"Rakib Real-ESRGAN API running on port {port}",
        flush=True
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=False
    )
