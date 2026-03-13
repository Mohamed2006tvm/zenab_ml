# -*- coding: utf-8 -*-
"""
Zenab ML Service — Legacy Edition (Python 2.7 compatible)
Provides simulated PM2.5 / PM10 analysis for the Zenab platform.
"""

import os
import json
import random
import BaseHTTPServer
import cgi

PORT = 8000

def pm25_to_aqi(pm25):
    bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
          (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)]
    for c_lo,c_hi,i_lo,i_hi in bp:
        if c_lo <= pm25 <= c_hi:
            return int(round(((i_hi-i_lo)/(c_hi-c_lo))*(pm25-c_lo)+i_lo))
    return 500

def aqi_status(aqi):
    if aqi<=50:  return "Good"
    if aqi<=100: return "Moderate"
    if aqi<=150: return "Unhealthy for Sensitive Groups"
    if aqi<=200: return "Unhealthy"
    if aqi<=300: return "Very Unhealthy"
    return "Hazardous"

def simulated_result(seed_val):
    random.seed(seed_val)
    pm25 = round(random.uniform(18, 245), 1)
    pm10 = round(pm25 * random.uniform(1.4, 2.2), 1)
    aqi  = pm25_to_aqi(pm25)
    conf = round(random.uniform(0.62, 0.94), 3)
    dets = random.randint(1, 9)
    random.seed()  # reset
    return {
        "pm25": pm25,
        "pm10": pm10,
        "aqi": aqi,
        "status": aqi_status(aqi),
        "confidence": conf,
        "detections": dets,
        "simulated": True,
        "note": "Running in Legacy Mode (Python 2.7) - Simulated data provided"
    }

class Handler(BaseHTTPServer.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to print to stdout so we can see it in terminal
        print("  " + self.address_string() + " - - [" + self.log_date_time_string() + "] " + (format % args))

    def send_json(self, code, data):
        body = json.dumps(data)
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
        print("📥 Received GET " + self.path)
        if self.path == "/" or self.path == "/health":
            self.send_json(200, {
                "service": "Zenab ML Service (Legacy)",
                "status": "running",
                "python_version": "2.7",
                "mode": "simulated"
            })
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        print("📥 Received POST " + self.path)
        if self.path != "/analyze":
            self.send_json(404, {"error": "Not found"})
            return

        try:
            # Simple content-length check for seed
            cl_header = self.headers.getheader('content-length')
            length = int(cl_header) if cl_header else 100
            
            # We don't strictly need to parse the image for simulation, 
            # but let's read the stream to avoid connection issues.
            self.rfile.read(length)
            
            result = simulated_result(length)
            print("✅ Returning simulated result for " + str(length) + " bytes")
            self.send_json(200, result)
        except Exception as e:
            print("❌ Error in POST: " + str(e))
            self.send_json(500, {"error": str(e)})

if __name__ == "__main__":
    server = BaseHTTPServer.HTTPServer(("0.0.0.0", PORT), Handler)
    print("\n🚀 Zenab ML Service (Legacy) running on http://localhost:" + str(PORT))
    print("   Mode: Simulated (Python 2.7 Compatibility)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
