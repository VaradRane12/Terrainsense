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

from flask import Flask, Response, jsonify, request, render_template
import threading
import camera
import recording
import sensor
import speech
import app_state
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
    return render_template("index.html")


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


@app.route("/audio/output/status", methods=["GET"])
def audio_output_status():
    st = speech.get_voice_status()
    return jsonify({"ok": True, "audio_output": st.get("audio_output", "unknown")}), 200


@app.route("/audio/output/bluetooth", methods=["GET", "POST"])
def audio_output_bluetooth():
    result = speech.set_audio_output("bluetooth")
    return jsonify(result), 200 if result.get("ok") else 409


@app.route("/audio/output/aux", methods=["GET", "POST"])
def audio_output_aux():
    result = speech.set_audio_output("aux")
    return jsonify(result), 200 if result.get("ok") else 409


@app.route("/video/stabilization/status", methods=["GET"])
def video_stabilization_status():
    return jsonify(camera.get_stabilization_status()), 200


@app.route("/video/stabilization/on", methods=["GET", "POST"])
def video_stabilization_on():
    return jsonify(camera.set_stabilization_enabled(True)), 200


@app.route("/video/stabilization/off", methods=["GET", "POST"])
def video_stabilization_off():
    return jsonify(camera.set_stabilization_enabled(False)), 200


# ── Sensor ────────────────────────────────────────────────────────────────────

@app.route("/sensor/status", methods=["GET"])
def sensor_status():
    return jsonify({"ok": True, "sensor": sensor.get_sensor_snapshot()}), 200


# ── App Control ────────────────────────────────────────────────────────────────

@app.route("/app/start", methods=["POST"])
def app_start():
    app_state.set_running(True)
    return jsonify({"ok": True, "message": "App started"}), 200


@app.route("/app/stop", methods=["POST"])
def app_stop():
    app_state.set_running(False)
    return jsonify({"ok": True, "message": "App stopped"}), 200


@app.route("/app/status", methods=["GET"])
def app_status():
    running = app_state.get_running()
    return jsonify({"ok": True, "running": running}), 200


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
    print("[INFO] Dashboard at http://<raspberry-pi-ip>:5000/")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        print("[INFO] Shutting down...")
