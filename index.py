import os
import io
import cv2
import numpy as np
import onnxruntime as ort

from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PORT = int(os.environ.get("PORT", 10000))

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "upscale_2x.onnx"
)

MAX_FILE_SIZE = 8 * 1024 * 1024
MAX_INPUT_SIZE = 1280


# =========================
# Load ONNX
# =========================

print("Loading lightweight ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions()
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX model loaded.")
print("Input:", input_name)
print("Output:", output_name)


# =========================
# Image processing
# =========================

def process_image(data):

    if not data:
        raise ValueError("Empty image")

    if len(data) > MAX_FILE_SIZE:
        raise ValueError("Image must be less than 8 MB")

    image = Image.open(io.BytesIO(data)).convert("RGB")

    w, h = image.size

    # Keep RAM low
    if max(w, h) > MAX_INPUT_SIZE:
        scale = MAX_INPUT_SIZE / max(w, h)

        w = int(w * scale)
        h = int(h * scale)

        image = image.resize(
            (w, h),
            Image.Resampling.LANCZOS
        )

    # RGB
    arr = np.asarray(image, dtype=np.float32)

    # NCHW / 0-1
    arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)

    # ONNX inference
    result = session.run(
        [output_name],
        {input_name: arr}
    )[0]

    # NCHW -> HWC
    result = np.squeeze(result)
    result = np.transpose(result, (1, 2, 0))

    result = np.clip(
        result * 255.0,
        0,
        255
    ).astype(np.uint8)

    output = Image.fromarray(
        result,
        "RGB"
    )

    buffer = io.BytesIO()

    output.save(
        buffer,
        format="JPEG",
        quality=92,
        optimize=True
    )

    buffer.seek(0)

    return buffer


# =========================
# Home
# =========================

@app.route("/")
def home():

    return jsonify({
        "service": "Rakib Lightweight ONNX Upscaler",
        "status": "online",
        "model": "lightweight-2x",
        "runtime": "ONNX Runtime CPU",
        "scale": "2x"
    })


# =========================
# Health
# =========================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model": "lightweight-2x",
        "runtime": "ONNX Runtime CPU"
    })


# =========================
# Upload
# =========================

@app.route("/upscale", methods=["POST"])
def upscale():

    try:

        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "message": "Use field name: image"
            }), 400

        file = request.files["image"]

        data = file.read()

        result = process_image(data)

        return send_file(
            result,
            mimetype="image/jpeg",
            download_name="upscaled.jpg"
        )

    except Exception as e:

        print("Upscale error:", repr(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================
# URL
# =========================

@app.route("/upscale-url", methods=["POST"])
def upscale_url():

    import requests

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
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        result = process_image(
            response.content
        )

        return send_file(
            result,
            mimetype="image/jpeg",
            download_name="upscaled.jpg"
        )

    except Exception as e:

        print("URL error:", repr(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================
# Start
# =========================

if __name__ == "__main__":

    print(
        f"Rakib Lightweight ONNX API running on port {PORT}"
    )

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=False
    )
