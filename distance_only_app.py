"""
Minimal Flask app: camera feed + ToF distance grid only.

Run:
    python3 distance_only_app.py
Open:
    http://<pi-ip>:5050/
"""

from flask import Flask, Response, jsonify, render_template_string

import app_state
import camera
import sensor

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Distance + Camera</title>
  <style>
    :root {
      --bg: #0b0f14;
      --panel: #131a22;
      --line: #243447;
      --text: #d9e2ec;
      --muted: #9fb3c8;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: radial-gradient(circle at top right, #1a2635, var(--bg) 45%);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
      padding: 14px;
    }

    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 14px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }

    .title {
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      letter-spacing: .5px;
      color: var(--muted);
      text-transform: uppercase;
    }

    .video img {
      width: 100%;
      display: block;
      background: #000;
    }

    .stats {
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      border-bottom: 1px solid var(--line);
    }

    .stat {
      background: #0e151d;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      min-height: 64px;
    }

    .k {
      color: var(--muted);
      font-size: 12px;
    }

    .v {
      font-size: 24px;
      line-height: 1.2;
      margin-top: 4px;
    }

    .grid {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(8, minmax(36px, 1fr));
      gap: 6px;
    }

    .cell {
      border-radius: 6px;
      border: 1px solid #2a3d52;
      height: 38px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      font-weight: 600;
      color: #091018;
      user-select: none;
    }

    .footer {
      padding: 0 12px 12px;
      color: var(--muted);
      font-size: 12px;
    }

    @media (max-width: 980px) {
      .wrap { grid-template-columns: 1fr; }
      .grid { grid-template-columns: repeat(8, minmax(30px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel video">
      <h2 class="title">Camera Feed</h2>
      <img src="/video_feed" alt="Camera feed">
    </section>

    <section class="panel">
      <h2 class="title">ToF Distance Grid</h2>
      <div class="stats">
        <div class="stat">
          <div class="k">Front Distance</div>
          <div class="v" id="front">-- cm</div>
        </div>
        <div class="stat">
          <div class="k">Valid Center Cells</div>
          <div class="v" id="valid">0</div>
        </div>
      </div>
      <div class="grid" id="grid"></div>
      <div class="footer" id="meta">Waiting for ToF data...</div>
    </section>
  </div>

  <script>
    const grid = document.getElementById("grid");
    for (let i = 0; i < 64; i++) {
      const d = document.createElement("div");
      d.className = "cell";
      d.textContent = "--";
      d.id = "c" + i;
      grid.appendChild(d);
    }

    function colorForCm(cm, valid) {
      if (!valid || cm <= 0) return "#4b5563";
      if (cm < 60) return "#e74c3c";
      if (cm < 120) return "#f39c12";
      if (cm < 220) return "#f1c40f";
      return "#2ecc71";
    }

    async function refresh() {
      try {
        const res = await fetch("/tof", { cache: "no-store" });
        const data = await res.json();

        document.getElementById("front").textContent = data.front_cm === null ? "-- cm" : `${data.front_cm} cm`;
        document.getElementById("valid").textContent = String(data.front_valid_cells || 0);

        const distances = data.distances || [];
        const status = data.status || [];
        for (let i = 0; i < 64; i++) {
          const mm = Number(distances[i] || 0);
          const st = Number(status[i] || 0);
          const valid = st === 5 && mm > 0;
          const cm = valid ? Math.round(mm / 10) : 0;
          const cell = document.getElementById("c" + i);
          cell.textContent = valid ? `${cm}` : "--";
          cell.style.background = colorForCm(cm, valid);
        }

        const t = data.last_update ? new Date(data.last_update * 1000).toLocaleTimeString() : "--";
        const err = data.error ? ` | error: ${data.error}` : "";
        document.getElementById("meta").textContent = `Updated: ${t}${err}`;
      } catch (e) {
        document.getElementById("meta").textContent = "ToF fetch failed";
      }
    }

    refresh();
    setInterval(refresh, 250);
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/video_feed")
def video_feed():
    return Response(camera.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/tof")
def tof_data():
    snap = sensor.get_sensor_snapshot()
    front_mm = snap.get("front_mm")
    front_cm = int(round(front_mm / 10.0)) if isinstance(front_mm, (int, float)) else None
    return jsonify(
        {
            "ok": True,
            "front_cm": front_cm,
            "front_valid_cells": int(snap.get("front_valid_cells", 0) or 0),
            "distances": list(snap.get("distances", [0] * 64))[:64],
            "status": list(snap.get("status", [0] * 64))[:64],
            "last_update": snap.get("last_update", 0.0),
            "error": snap.get("error"),
        }
    )


if __name__ == "__main__":
    app_state.set_running(False)
    sensor.start_sensor_thread()
    camera.start_pipeline()

    print("[INFO] Distance-only app at http://0.0.0.0:5050/")
    app.run(host="0.0.0.0", port=5050, debug=False)
