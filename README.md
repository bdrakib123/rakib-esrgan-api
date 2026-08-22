# 🚀 Rakib Real-ESRGAN API

AI-powered image upscaling API using Real-ESRGAN.

Enhance low-resolution images and generate high-quality 4× upscaled images with deep learning.

## ✨ Features

 AI-powered image upscaling  
 RealESRGAN_x4plus model  
 4× image enhancement  
 CPU support  
 REST API  
 Upload image directly  
 Upscale image from URL  
 CORS enabled  
 Health check endpoint  
 Render deployment ready  

## 🔥 API Endpoints

### API Status

GET /

Example response:

{
  "service": "Rakib Real-ESRGAN API",
  "status": "online",
  "model": "RealESRGAN_x4plus",
  "scale": "4x",
  "device": "CPU"
}

### Health Check

GET /health

Returns the current API and model status.

### Upload & Upscale

POST /upscale

Upload an image using multipart/form-data.

Example:

curl -X POST \
  -F "image=@image.jpg" \
  http://localhost:5000/upscale \
  --output result.png

### Upscale From URL

POST /upscale-url

Example:

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' \
  http://localhost:5000/upscale-url \
  --output result.png

## 🧠 Model

Model: RealESRGAN_x4plus

Upscaling: 4×

Real-ESRGAN is a deep-learning based image restoration and super-resolution model.

## 🛠️ Technologies

Python  
Flask  
PyTorch  
TorchVision  
OpenCV  
Pillow  
BasicSR  
Real-ESRGAN  

## 📂 Project Structure

rakib-esrgan-api/

 index.py
 requirements.txt
 render.yaml
 README.md
 LICENSE

 weights/
   └── RealESRGAN_x4plus.pth

 basicsr-patched/

## 💻 Local Setup

Clone the repository:

git clone https://github.com/hoon6t9/rakib-esrgan-api.git

Enter the project:

cd rakib-esrgan-api

Create virtual environment:

python3 -m venv venv

Activate:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Start the API:

python index.py

Default local address:

http://127.0.0.1:5000

## ☁️ Render Deployment

This project includes Render configuration.

Build command:

pip install --no-build-isolation ./basicsr-patched && pip install -r requirements.txt

Start command:

python index.py

The application automatically uses the PORT environment variable provided by Render.

## ⚡ Performance

Real-ESRGAN is computationally intensive.

CPU processing may take longer for large images.

GPU hardware is recommended for faster processing.

## 👨‍💻 Author

Rakib Hasan

Built with ❤️ for AI-powered image enhancement.

## 📄 License

This project is licensed under the MIT License.

See LICENSE for details.

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ star.

