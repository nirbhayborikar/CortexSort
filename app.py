#!/usr/bin/env python3
"""
CortexSort Web UI — Chat-driven robotic manipulation with live simulation view.

Left panel: ChatGPT-style chat window for natural language commands.
Right panel: Live MJPEG stream from the PyBullet simulation camera.
"""

import io
import json
import threading
import time

import numpy as np
import pybullet as p
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image

from cortex_engine import VisionLanguagePandaSimulation

app = Flask(__name__)

sim: VisionLanguagePandaSimulation = None
sim_lock = threading.Lock()
latest_frame: bytes = b""
sim_ready = threading.Event()


def init_simulation():
    """Boot PyBullet + CLIP in headless mode and warm up the scene."""
    global sim
    print("[*] Initializing simulation (headless) + CLIP model...")
    sim = VisionLanguagePandaSimulation(gui_mode=False, model_name="openai/clip-vit-base-patch32")
    sim.setup_simulation()

    for _ in range(240):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    sim.set_initial_robot_pose()
    sim.frame_callback = _update_frame  # live frame updates during robot movement
    _update_frame()
    sim_ready.set()
    print("[+] Simulation ready.")


def _capture_jpeg() -> bytes:
    """Render the current PyBullet camera view as JPEG bytes."""
    pil_img = sim.capture_camera_image()
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _update_frame():
    """Snapshot the sim camera into the shared frame buffer."""
    global latest_frame
    latest_frame = _capture_jpeg()


def frame_streamer():
    """Background thread that updates the MJPEG frame when sim is idle."""
    sim_ready.wait()
    while True:
        acquired = sim_lock.acquire(timeout=0.05)
        if acquired:
            try:
                _update_frame()
            finally:
                sim_lock.release()
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/video_feed")
def video_feed():
    """MJPEG stream for the right-panel simulation view."""
    def generate():
        while True:
            frame = latest_frame
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.1)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process a natural-language command and drive the robot."""
    data = request.get_json(force=True)
    user_msg = (data.get("message") or "").strip()

    if not user_msg:
        return jsonify({"reply": "Please type a command, e.g. *pick the red cube*."})

    sim_ready.wait()

    with sim_lock:
        try:
            result = _handle_command(user_msg)
            _update_frame()
        except Exception as exc:
            result = {"reply": f"Error: {exc}"}

    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset the simulation to its initial state."""
    sim_ready.wait()
    with sim_lock:
        p.resetSimulation()
        sim.objects = {}
        sim.setup_simulation()
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
        sim.set_initial_robot_pose()
        _update_frame()
    return jsonify({"reply": "Simulation reset. All objects restored to their starting positions."})


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

def _handle_command(text: str) -> dict:
    """Parse a user message, run CLIP matching + robot action, return reply dict."""

    lower = text.lower()

    # --- auto sort shortcut ---
    if "sort" in lower and ("all" in lower or "auto" in lower or "color" in lower):
        results = sim.run_color_sorting_demo(visualize=False)
        ok = sum(1 for r in results if r["success"])
        lines = [f"Sorted **{ok}/{len(results)}** objects by color."]
        for r in results:
            tag = "done" if r["success"] else "fail"
            lines.append(f"  [{tag}] {r['object_name']} -> {r.get('target_zone', '?')}")
        return {"reply": "\n".join(lines)}

    # --- single object pick via CLIP ---
    camera_image, bboxes, crops = sim.capture_and_process_scene()

    if not bboxes:
        return {"reply": "No objects detected in the scene."}

    scores = sim.compute_object_similarity(crops, text)
    if not scores:
        return {"reply": "Could not compute similarity — try a different prompt."}

    selected, best_score = sim.select_best_object(scores)
    if not selected:
        return {"reply": "No matching object found."}

    score_lines = []
    for name, sc in sorted(scores.items(), key=lambda x: -x[1]):
        bar = int(sc * 20) * "█"
        marker = " <--" if name == selected else ""
        score_lines.append(f"  {name:18s} {sc:.3f} {bar}{marker}")

    # Determine target zone from command or default to color-match
    zone = _parse_zone(lower)

    success = sim.pick_object(selected)
    if not success:
        return {"reply": f"Failed to pick up **{selected}**."}

    if zone:
        _place_in_zone(selected, zone)
        zone_label = zone.replace("_", " ").title()
        action_msg = f"Placed in **{zone_label}**."
    else:
        sim.place_object_in_sorting_zone(selected)
        auto_zone = sim._get_target_zone_for_object(selected)
        zone_label = auto_zone.replace("_", " ").title()
        action_msg = f"Auto-placed in **{zone_label}** (color match)."

    reply = (
        f"**CLIP scores** for *\"{text}\"*:\n"
        + "\n".join(score_lines)
        + f"\n\nSelected **{selected}** (confidence {best_score:.3f}).\n"
        + action_msg
    )
    return {"reply": reply}


def _parse_zone(text: str):
    """Extract an explicit zone from the user message, or None."""
    for color in ("red", "blue", "green", "yellow"):
        if f"{color} zone" in text or f"to {color}" in text or f"in {color}" in text:
            return f"{color}_zone"
    if "throw" in text:
        return "__throw__"
    return None


def _place_in_zone(obj_name: str, zone: str):
    """Place held object into the requested zone."""
    if zone == "__throw__":
        # quick toss
        sim.move_to_position([-0.3, 0.5, 0.8])
        sim.control_gripper(sim.gripper_open_position)
        sim.move_to_position(sim.home_position)
        return

    if zone not in sim.sorting_zones:
        sim.place_object_in_sorting_zone(obj_name)
        return

    zc = sim.sorting_zones[zone]
    zp = zc["position"]
    sim.safe_move_to_position([zp[0], zp[1], zp[2] + 0.25])
    sim.move_to_position([zp[0], zp[1], zp[2] + 0.05])
    sim.control_gripper(sim.gripper_open_position)
    sim.move_to_position([zp[0], zp[1], zp[2] + 0.25])
    sim.move_to_position(sim.home_position)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Boot sim in background so Flask starts fast
    threading.Thread(target=init_simulation, daemon=True).start()
    threading.Thread(target=frame_streamer, daemon=True).start()

    print("[*] Starting CortexSort web UI on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
