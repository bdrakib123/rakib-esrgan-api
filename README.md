# 🚀 Rakib Real-ESRGAN API

AI-powered image upscaling API built with Python, Flask, PyTorch and Real-ESRGAN.

Enhance low-resolution images and generate high-quality 4x upscaled images.

## ✨ Features

- 🧠 Real-ESRGAN AI upscaling
- 🔍 4x image enhancement
- 🖼️ Image upload support
- 🌐 Image URL support
- ⚡ REST API
- 🐍 Python + Flask
- 🖥️ CPU compatible
- ☁️ Render ready
- 🌍 CORS enabled
- 📱 Easy integration

## 📡 API Endpoints

### GET /

Returns API information.

### GET /health

Checks API status.

### POST /upscale

Upload an image and receive a 4x upscaled PNG.

Example:

curl -X POST \
  -F "image=@image.jpg" \
  https://YOUR-APP.onrender.com/upscale \
  --output upscaled.png

### POST /upscale-url

Upscale an image from a URL.

Example:

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' \
  https://YOUR-APP.onrender.com/upscale-url \
  --output upscaled.png

## 🛠️ Technology

- Python
- Flask
- PyTorch
- Real-ESRGAN
- BasicSR
- OpenCV
- Pillow
- Flask-CORS

## 📁 Project Structure

rakib-esrgan-api/
├── index.py
├── requirements.txt
├── render.yaml
├── README.md
├── .gitignore
└── weights/
    └── RealESRGAN_x4plus.pth

## 🚀 Render Deployment

Build Command:

pip install -r requirements.txt

Start Command:

python index.py

The API automatically uses the PORT provided by Render.

## 💻 Local Installation

Clone the repository:

git clone https://github.com/bdrakib123/rakib-esrgan-api.git

Enter the project:

cd rakib-esrgan-api

Create virtual environment:

python3 -m venv venv

Activate:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run:

python index.py

## 🧠 Model

Model: RealESRGAN_x4plus

Upscaling: 4x

Device: CPU

## ⚠️ Notes

- Maximum upload size is 20 MB.
- Processing speed depends on image resolution.
- CPU processing is slower than GPU.
- Large images may require more RAM.
- Render free instances may take some time to wake up.

## 👨‍💻 Author

Rakib Hasan

GitHub:
https://github.com/bdrakib123

## ⭐ Support

If you find this project useful, consider giving the repository a star.

## 📜 License

This project is for educational and development purposes.

Real-ESRGAN is developed by its original authors.
