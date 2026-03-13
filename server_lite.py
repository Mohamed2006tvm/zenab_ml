"""
Zenab ML Service — Lightweight Edition
Zero external dependencies. Uses only Python built-ins.
When Best.pt + ultralytics are available it runs real YOLO inference.
Otherwise, returns simulated PM2.5 / PM10 data.
Run:  python3 server_lite.py
"""

import os
import io
import cgi
import json
import math
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Best.pt")

# ── Try to load YOLO model (optional) ────────────────────────────────────────
model = None
def try_load_model():
    global model
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Model not loaded ({e}) — will return simulated data")

# ── AQI helpers ───────────────────────────────────────────────────────────────
def pm25_to_aqi(pm25):
    bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
          (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)]
    for c_lo,c_hi,i_lo,i_hi in bp:
        if c_lo <= pm25 <= c_hi:
            return round(((i_hi-i_lo)/(c_hi-c_lo))*(pm25-c_lo)+i_lo)
    return 500

def aqi_status(aqi):
    if aqi<=50:  return "Good"
    if aqi<=100: return "Moderate"
    if aqi<=150: return "Unhealthy for Sensitive Groups"
    if aqi<=200: return "Unhealthy"
    if aqi<=300: return "Very Unhealthy"
    return "Hazardous"

# ── Real inference ────────────────────────────────────────────────────────────
def run_inference(image_bytes):
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model(img, verbose=False)
        pm25_confs, pm10_confs = [], []
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                (pm25_confs if cls==0 else pm10_confs).append(conf)
        pm25 = round(float(sum(pm25_confs)/len(pm25_confs))*300,1) if pm25_confs else 0.0
        pm10 = round(float(sum(pm10_confs)/len(pm10_confs))*600,1) if pm10_confs else 0.0
        conf = round(float(sum(pm25_confs+pm10_confs)/max(len(pm25_confs+pm10_confs),1)),3)
        aqi  = pm25_to_aqi(pm25)
        return {"pm25":pm25,"pm10":pm10,"aqi":aqi,"status":aqi_status(aqi),
                "confidence":conf,"detections":len(pm25_confs)+len(pm10_confs),"simulated":False}
    except Exception as e:
        raise RuntimeError(f"Inference error: {e}")

# ── Simulated data (when model not loaded) ────────────────────────────────────
def simulated_result(image_bytes):
    # Use image size as seed for reproducible results per image
    seed = len(image_bytes)
    random.seed(seed)
    pm25 = round(random.uniform(18, 245), 1)
    pm10 = round(pm25 * random.uniform(1.4, 2.2), 1)
    aqi  = pm25_to_aqi(pm25)
    conf = round(random.uniform(0.62, 0.94), 3)
    dets = random.randint(1, 9)
    random.seed()  # reset
    return {"pm25":pm25,"pm10":pm10,"aqi":aqi,"status":aqi_status(aqi),
            "confidence":conf,"detections":dets,"simulated":True,
            "note":"Best.pt not loaded — place it in MLService/ for real readings"}

# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"  {self.address_string()} {format % args}")

    def send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/health":
            self.send_json(200, {
                "service": "Zenab ML Service (lite)",
                "status": "running",
                "model_loaded": model is not None,
                "model_path": MODEL_PATH,
                "port": PORT
            })
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/analyze":
            self.send_json(404, {"error": "Not found"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json(400, {"error": "Expected multipart/form-data"})
            return

        # Parse multipart body
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        # Extract boundary
        boundary = None
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[9:].strip('"').encode()
                break

        if not boundary:
            self.send_json(400, {"error": "No boundary in multipart"})
            return

        image_bytes = self._extract_image(body, boundary)
        if image_bytes is None:
            self.send_json(400, {"error": "No image data found in request"})
            return

        try:
            if model is not None:
                result = run_inference(image_bytes)
            else:
                result = simulated_result(image_bytes)
            self.send_json(200, result)
        except Exception as e:
            self.send_json(500, {"error": str(e)})

    def _extract_image(self, body, boundary):
        """Pull the first binary file part from a multipart body."""
        delim = b"--" + boundary
        parts = body.split(delim)
        for part in parts:
            if b"Content-Disposition" not in part:
                continue
            if b'filename="' not in part and b"filename*=" not in part:
                continue
            # Split headers from body at double CRLF
            idx = part.find(b"\r\n\r\n")
            if idx == -1:
                idx = part.find(b"\n\n")
                if idx == -1:
                    continue
                data = part[idx+2:]
            else:
                data = part[idx+4:]
            # Strip trailing boundary markers / CRLF
            data = data.rstrip(b"\r\n-")
            if data:
                return data
        return None


if __name__ == "__main__":
    try_load_model()
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\n🚀 Zenab ML Service running on http://localhost:{PORT}")
    print(f"   Model: {'✅ Loaded — Best.pt' if model else '⚠️  Simulated mode (no Best.pt)'}")
    print(f"   POST  /analyze  — upload image, get PM2.5 / PM10 / AQI")
    print(f"   GET   /         — health check\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
