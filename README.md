# 🔍 Want-That: Find Products Through Images

## 🎯 What is Want-That?
Hey there! Want-That is a cool API that helps you find products online just by uploading a photo. See something you like? Snap! 📸 Upload the image and we'll help you find similar products.

## ✨ Features
- 🖼️ Visual product search
- 🤖 AI-powered image analysis
- 🛍️ Online price comparison
- 🚀 Fast and efficient responses
- 🎨 Visual feature detection

## 🛠️ Tech Stack
- FastAPI
- Python 3.8+
- AI models for image analysis
- Price comparison APIs

## 🚀 Getting Started

### 📦 Installation

1. Clone and enter the repo:

git clone https://github.com/username/Want-That-api.git
cd Want-That-api

2. Install dependencies:

pip install fastapi uvicorn

3. Let's run! 🏃‍♂️

uvicorn src.app:app --reload --port 8000

## 🎮 How to Use

1. Start the server: `http://localhost:8000`
2. 📸 Upload an image of the product you're interested in
3. 🔍 The API analyzes the image and searches for similar products
4. 💫 Voilà! You'll get a list of similar products

## 📚 API Documentation
- 🎯 Swagger UI: `http://localhost:8000/docs`
- 📖 ReDoc: `http://localhost:8000/redoc`

## 🔥 Main Endpoints

### Image Search
- `POST /api/search` - Upload an image to find similar products
- `GET /api/results/{search_id}` - Get search results

## 👩‍💻 Want to Contribute?
Awesome! We love help. Here's how:

1. 🍴 Fork the repo
2. 🌱 Create your branch (`git checkout -b feature/super-feature`)
3. 💪 Make your changes
4. 🚀 Push to the branch (`git push origin feature/super-feature`)
5. 🎉 Open a Pull Request

## 🐛 Found a Bug?
No worries! Open an issue and we'll look into it together 🤝

## 📝 License
MIT - Use it as you wish! 🎉

## 🤝 Need Help?
- 📧 Email: your@email.com
- 🐦 Twitter: @youruser
- 💬 Discord: [Join our community]

## 🚀 Coming Soon
- [ ] Multi-image search
- [ ] Advanced search filters
- [ ] More online store integrations
- [ ] Companion mobile app

---
Made with ❤️ by Alejandro Candela