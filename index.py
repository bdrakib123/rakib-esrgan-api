import os
import io
import gc
import requests

from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MAX_FILE_SIZE = 10 * 1024 * 1024


def process_image(data):
    if not data:
        raise ValueError("Empty image")

    if len(data) > MAX_FILE_SIZE:
        raise ValueError("Image size must be less than 10 MB")

    image = Image.open(io.BytesIO(data)).convert("RGB")

    # Prevent huge RAM usage
    max_size = 2048

    if max(image.size) > max_size:
        ratio = max_size / max(image.size)

        image = image.resize(
            (
                int(image.width * ratio),
                int(image.height * ratio)
            ),
            Image.Resampling.LANCZOS
        )

    # Lightweight 2x upscale
    new_size = (
        image.width * 2,
        image.height * 2
    )

    image = image.resize(
        new_size,
        Image.Resampling.LANCZOS
    )

    # Small sharpening
    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=1,
            percent=80,
            threshold=3
        )
    )

    # Slight contrast enhancement
    image = ImageEnhance.Contrast(
        image
    ).enhance(1.03)

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=92,
        optimize=True
    )

    output.seek(0)

    image.close()
    gc.collect()

    return output


@app.route("/")
def home():
    return jsonify({
        "service": "Rakib Image Upscaler API",
        "status": "online",
        "model": "Lightweight CPU",
        "scale": "2x",
        "device": "CPU",
        "endpoints": {
            "health": "/health",
            "upscale": "POST /upscale",
            "upscale_url": "POST /upscale-url"
        }
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Rakib Image Upscaler API",
        "device": "CPU"
    })


@app.route("/upscale", methods=["POST"])
def upscale():

    try:
        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "message": "Use image field"
            }), 400

        file = request.files["image"]

        if not file.filename:
            return jsonify({
                "status": "error",
                "message": "Invalid filename"
            }), 400

        data = file.read()

        result = process_image(data)

        return send_file(
            result,
            mimetype="image/jpeg",
            download_name="upscaled.jpg"
        )

    except Exception as e:

        gc.collect()

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

        if len(response.content) > MAX_FILE_SIZE:
            return jsonify({
                "status": "error",
                "message": "Image too large"
            }), 400

        result = process_image(
            response.content
        )

        return send_file(
            result,
            mimetype="image/jpeg",
            download_name="upscaled.jpg"
        )

    except Exception as e:

        gc.collect()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    print(
        f"Rakib Image Upscaler running on port {port}",
        flush=True
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=False
    )
