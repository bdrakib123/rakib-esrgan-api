import os
import io
import requests
import numpy as np
import onnxruntime as ort

from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

MAX_FILE_SIZE = 20 * 1024 * 1024
SCALE = 4

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "realesr-general-x4v3.onnx"
)


print("Loading ONNX model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

INPUT_NAME = input_info.name
OUTPUT_NAME = output_info.name

print("ONNX model loaded successfully.")
print("Input:", INPUT_NAME)
print("Output:", OUTPUT_NAME)


def run_model(image_bytes):

    if not image_bytes:
        raise ValueError("Empty image")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise ValueError("Image size must be less than 20 MB")

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    original_w, original_h = image.size

    img = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # RGB HWC -> NCHW
    img = np.transpose(
        img,
        (2, 0, 1)
    )

    img = np.expand_dims(
        img,
        axis=0
    )

    # Run ONNX
    result = session.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME: img
        }
    )[0]

    # NCHW -> HWC
    result = result[0]

    result = np.transpose(
        result,
        (1, 2, 0)
    )

    result = np.clip(
        result,
        0,
        1
    )

    result = (
        result * 255.0
    ).astype(np.uint8)

    output = Image.fromarray(
        result,
        "RGB"
    )

    # Safety: make sure exact 4x size
    expected_size = (
        original_w * SCALE,
        original_h * SCALE
    )

    if output.size != expected_size:
        output = output.resize(
            expected_size,
            Image.Resampling.LANCZOS
        )

    buffer = io.BytesIO()

    output.save(
        buffer,
        format="JPEG",
        quality=95,
        optimize=True
    )

    buffer.seek(0)

    return buffer


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "service": "Rakib Real-ESRGAN ONNX API",
        "status": "online",
        "model": "realesr-general-x4v3",
        "scale": "4x",
        "runtime": "ONNX Runtime CPU",
        "endpoints": {
            "health": "/health",
            "upscale": "POST /upscale",
            "upscale_url": "POST /upscale-url"
        }
    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model": "realesr-general-x4v3",
        "runtime": "ONNX Runtime CPU"
    })


@app.route("/upscale", methods=["POST"])
def upscale():

    try:

        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image uploaded. Use field name: image"
            }), 400

        file = request.files["image"]

        image_bytes = file.read()

        result = run_model(
            image_bytes
        )

        return send_file(
            result,
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="upscaled.jpg"
        )

    except Exception as e:

        print("Upscale error:", repr(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


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

        result = run_model(
            response.content
        )

        return send_file(
            result,
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="upscaled.jpg"
        )

    except Exception as e:

        print(
            "URL upscale error:",
            repr(e)
        )

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        f"Rakib Real-ESRGAN ONNX API running on port {port}"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
