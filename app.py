import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ─── Load YOLO model ──────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "Best.pt"))
model = None

def load_model():
    global model
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        model = None

# ─── AQI calculation from PM2.5 ───────────────────────────────────────────────
def pm25_to_aqi(pm25):
    """Convert PM2.5 concentration (µg/m³) to AQI using EPA breakpoints."""
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,   51,  100),
        (35.5,  55.4,   101, 150),
        (55.5,  150.4,  151, 200),
        (150.5, 250.4,  201, 300),
        (250.5, 350.4,  301, 400),
        (350.5, 500.4,  401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
            return round(aqi)
    return 500

def aqi_status(aqi):
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"

# ─── Inference ────────────────────────────────────────────────────────────────
def run_inference(image_bytes):
    """
    Run YOLO inference on the image.
    The model is expected to output bounding boxes where:
      - class 0 = PM2.5 particle region
      - class 1 = PM10 particle region
    Confidence scores are mapped to concentration estimates.
    """
    if model is None:
        raise RuntimeError("Model not loaded")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(img, verbose=False)

    pm25_detections = []
    pm10_detections = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            # Map confidence → concentration estimate
            # Assumes model confidence correlates with particle density
            if cls == 0:  # PM2.5
                pm25_detections.append(conf)
            elif cls == 1:  # PM10
                pm10_detections.append(conf)

    # Convert confidence to µg/m³ concentrations
    # Scale: confidence 0-1 → PM2.5 0-300 µg/m³, PM10 0-600 µg/m³
    pm25 = round(float(np.mean(pm25_detections)) * 300, 1) if pm25_detections else 0.0
    pm10 = round(float(np.mean(pm10_detections)) * 600, 1) if pm10_detections else 0.0

    # Use detected count as confidence proxy
    confidence = round(
        float(np.mean(pm25_detections + pm10_detections)) if (pm25_detections or pm10_detections) else 0.0,
        3
    )
    detections = len(pm25_detections) + len(pm10_detections)

    aqi = pm25_to_aqi(pm25)
    status = aqi_status(aqi)

    return {
        "pm25": pm25,
        "pm10": pm10,
        "aqi": aqi,
        "status": status,
        "confidence": confidence,
        "detections": detections,
    }

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "service": "Zenab ML Service",
        "status": "running",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accept an image upload, run YOLO inference, return PM2.5/PM10/AQI.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    image_bytes = file.read()

    if model is None:
        # Return simulated data if model not loaded (for dev/testing)
        import random
        pm25 = round(random.uniform(15, 280), 1)
        pm10 = round(pm25 * random.uniform(1.4, 2.1), 1)
        aqi  = pm25_to_aqi(pm25)
        return jsonify({
            "pm25": pm25,
            "pm10": pm10,
            "aqi": aqi,
            "status": aqi_status(aqi),
            "confidence": round(random.uniform(0.6, 0.95), 3),
            "detections": random.randint(1, 8),
            "simulated": True,
            "note": "Model not loaded — returning simulated data",
        })

    try:
        result = run_inference(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 ML Service running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
