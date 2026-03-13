#!/bin/bash
# ============================================================
# Zenab ML Service Setup Script
# Run this ONCE to set up the Python environment
# ============================================================
echo "🔧 Setting up Zenab ML Service..."

# Step 1: Install required system packages
echo "📦 Installing system packages (requires sudo)..."
sudo apt install -y python3-pip python3-venv python3-dev 2>/dev/null || \
  sudo apt-get install -y python3-pip python3-venv python3-dev

# Step 2: Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Step 3: Activate and install packages
echo "📥 Installing Python dependencies..."
source venv/bin/activate
pip install flask flask-cors pillow ultralytics opencv-python-headless numpy requests

echo ""
echo "✅ Setup complete!"
echo ""
echo "📌 NEXT STEPS:"
echo "   1. Copy Best.pt into this folder:  /home/mohamed/Desktop/ZENAB/MLService/"
echo "   2. Start the ML service:            source venv/bin/activate && python3 app.py"
echo "   3. Start the backend:               cd ../Backend && npm run dev"
echo "   4. Start the frontend:              cd ../FrontEnd && npm run dev"
echo "   5. Open browser:                    http://localhost:5174"
echo "   6. Navigate to:                     /analyze  (after logging in)"
