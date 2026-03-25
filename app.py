"""
app.py
──────
Flask entry point and HTTP routes.
All business logic lives in the other modules — this file is intentionally thin.

Startup sequence
────────────────
  1. sensor.start_sensor_thread()      ToF / IMU subprocess reader
  2. speech.start_speech_thread()      espeak worker
  3. recording.start_recording_thread() disk writer
  4. recording.register_with_speech()  wire speech events → audio log
  5. camera.start_pipeline()           Picamera2 capture + TFLite inference
  6. app.run()                          Flask HTTP server
"""

from flask import Flask, Response, jsonify, request

import camera
import recording
import sensor
import speech
from config import (
    DEFAULT_RECORD_FPS,
    DEFAULT_RECORD_MINUTES,
    ENABLE_SENSOR_BRIDGE,
    ENABLE_VOICE,
    VOICE_INTERVAL_SEC,
    VOICE_TEST_TEXT,
)

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return (
        "<html><body style='background:#111;text-align:center;margin:0;padding:20px;'>"
        "<h1 style='color:#00ff99;font-family:Arial;'>TerrainSense Live</h1>"
        "<div style='margin-bottom:12px;color:#d5ffe8;font-family:Arial;'>"
        "Minutes: <input id='mins' type='number' min='1' value='5' style='width:70px;'>"
        "<button onclick='startRec()' style='margin-left:8px;padding:8px 12px;'>Start Recording</button>"
        "<button onclick='stopRec()' style='margin-left:8px;padding:8px 12px;'>Stop</button>"
        "<button onclick='voiceOn()' style='margin-left:8px;padding:8px 12px;'>Voice ON</button>"
        "<button onclick='voiceOff()' style='margin-left:8px;padding:8px 12px;'>Voice OFF</button>"
        "<button onclick='voiceTest()' style='margin-left:8px;padding:8px 12px;'>Voice TEST</button>"
        "<div id='recStatus' style='margin-top:8px;font-size:14px;'></div>"
        "<div id='voiceStatus' style='margin-top:4px;font-size:14px;'></div>"
        "</div>"
        "<img src='/video_feed' style='width:100%;max-width:860px;border:2px solid #00ff99;border-radius:8px;'>"
        "<script>"
        "async function refreshStatus(){"
        "  const r=await fetch('/record/status');const j=await r.json();"
        "  document.getElementById('recStatus').textContent=j.recording"
        "    ?`Recording ON | ${j.seconds_left}s left | File: ${j.saved_to}`:'Recording OFF';"
        "  const vr=await fetch('/voice/status');const vj=await vr.json();"
        "  document.getElementById('voiceStatus').textContent=vj.voice_enabled?'Voice ON':'Voice OFF';"
        "}"
        "async function startRec(){"
        "  const m=document.getElementById('mins').value||'5';"
        "  const r=await fetch(`/record/start?minutes=${encodeURIComponent(m)}`);"
        "  const j=await r.json();"
        "  document.getElementById('recStatus').textContent=j.error||('Started: '+j.saved_to);"
        "}"
        "async function stopRec(){"
        "  const r=await fetch('/record/stop');const j=await r.json();"
        "  document.getElementById('recStatus').textContent=j.saved_to?('Saved: '+j.saved_to):'Stopped';"
        "}"
        "async function voiceOn(){await fetch('/voice/on');refreshStatus();}"
        "async function voiceOff(){await fetch('/voice/off');refreshStatus();}"
        "async function voiceTest(){"
        "  const r=await fetch('/voice/test');const j=await r.json();"
        "  document.getElementById('voiceStatus').textContent=j.message||j.error||'Voice test sent';"
        "}"
        "setInterval(refreshStatus,1000);refreshStatus();"
        "</script></body></html>"
    )


@app.route("/video_feed")
def video_feed():
    return Response(camera.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Recording ─────────────────────────────────────────────────────────────────

@app.route("/record/start", methods=["GET", "POST"])
def record_start():
    try:
        minutes = float(request.args.get("minutes", DEFAULT_RECORD_MINUTES))
        fps     = float(request.args.get("fps",     DEFAULT_RECORD_FPS))
        result  = recording.start_recording(minutes=minutes, fps=fps)
        return jsonify(result), 200 if result.get("ok") else 409
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/record/stop", methods=["GET", "POST"])
def record_stop():
    return jsonify(recording.stop_recording()), 200


@app.route("/record/status", methods=["GET"])
def record_status():
    return jsonify(recording.get_record_status()), 200


# ── Voice ─────────────────────────────────────────────────────────────────────

@app.route("/voice/status", methods=["GET"])
def voice_status():
    return jsonify(speech.get_voice_status()), 200


@app.route("/voice/on", methods=["GET", "POST"])
def voice_on():
    return jsonify(speech.set_voice_enabled(True)), 200


@app.route("/voice/off", methods=["GET", "POST"])
def voice_off():
    return jsonify(speech.set_voice_enabled(False)), 200


@app.route("/voice/test", methods=["GET", "POST"])
def voice_test():
    status = speech.get_voice_status()
    if not status["voice_enabled"]:
        return jsonify({"ok": False, "error": "Voice is OFF. Turn voice on first."}), 409
    speech.queue_speech(VOICE_TEST_TEXT)
    return jsonify({"ok": True, "message": f"Queued voice test: {VOICE_TEST_TEXT}"}), 200


# ── Sensor ────────────────────────────────────────────────────────────────────

@app.route("/sensor/status", methods=["GET"])
def sensor_status():
    return jsonify({"ok": True, "sensor": sensor.get_sensor_snapshot()}), 200


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Sensor bridge (ToF + IMU)
    sensor.start_sensor_thread()

    # 2. Speech worker
    speech.start_speech_thread()
    if ENABLE_VOICE:
        print(f"[INFO] Voice guidance: enabled (min interval {VOICE_INTERVAL_SEC:.1f}s)")
    else:
        print("[INFO] Voice guidance: disabled")

    # 3. Recording writer thread + wire speech events
    recording.start_recording_thread()
    recording.register_with_speech()

    # 4. Camera capture + inference pipeline
    camera.start_pipeline()

    print("[INFO] Open browser at  http://<raspberry-pi-ip>:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
