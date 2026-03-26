"""
app_hotspot.py
--------------
Headless hotspot runner:
- No Flask / web server
- Starts sensor + inference + voice
- Starts recording immediately
- Stops and finalizes recording on Ctrl+C

Run:
    python3 app_hotspot.py
"""

import os
import time

import app_state
import camera
import recording
import sensor
import speech
from config import ENABLE_VOICE, VOICE_INTERVAL_SEC


# A long session by default so recording only stops on Ctrl+C.
HEADLESS_RECORD_MINUTES = float(os.environ.get("HEADLESS_RECORD_MINUTES", "720"))


def _print_status_loop() -> None:
    """Periodic console status so user can verify runtime health without a web UI."""
    last_print = 0.0
    while True:
        now = time.time()
        if now - last_print >= 2.0:
            snap = sensor.get_sensor_snapshot()
            rec = recording.get_record_status()
            front = snap.get("front_mm")
            front_cm = "--" if front is None else f"{front / 10.0:.1f}"
            print(
                f"[STATUS] tof_front={front_cm}cm valid={snap.get('front_valid_cells', 0)} "
                f"recording={rec.get('recording', False)} left={rec.get('seconds_left', 0)}s"
            )
            last_print = now
        time.sleep(0.2)


if __name__ == "__main__":
    app_state.set_running(True)

    try:
        sensor.start_sensor_thread()
        speech.start_speech_thread()
        if ENABLE_VOICE:
            print(f"[INFO] Voice guidance: enabled (min interval {VOICE_INTERVAL_SEC:.1f}s)")
        else:
            print("[INFO] Voice guidance: disabled")

        recording.start_recording_thread()
        recording.register_with_speech()
        camera.start_pipeline()

        start_result = recording.start_recording(minutes=HEADLESS_RECORD_MINUTES)
        if not start_result.get("ok"):
            raise RuntimeError(start_result.get("error", "Failed to start recording"))

        print("[INFO] Headless hotspot mode running (no Flask).")
        print(f"[INFO] Recording to: {start_result.get('saved_to')}")
        print("[INFO] Press Ctrl+C to stop and finalize recording.")

        _print_status_loop()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Stopping...")
    except Exception as exc:
        print(f"[ERROR] Headless startup/runtime failure: {exc}")
        print("[HINT] Stop other camera users first (for example app.py).")
    finally:
        try:
            stop_result = recording.stop_recording()
            print(f"[INFO] Recording stopped: {stop_result.get('saved_to')}")
        except Exception as exc:
            print(f"[WARN] Could not stop recording cleanly: {exc}")
        try:
            camera.stop_pipeline()
        except Exception:
            pass
