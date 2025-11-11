# app.py
import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import os
import matplotlib.pyplot as plt

from detection import LaneDetector
from controller import FourWayController, FixedController
from siren import detect_siren_from_audio_bytes

st.set_page_config(layout="wide", page_title="AI Traffic Management System")
st.title("AI Traffic Management — Adaptive vs Fixed (Demo-ready)")

# Sidebar controls
st.sidebar.header("Configuration")
controller_type = st.sidebar.selectbox("Controller Type", ["Adaptive AI", "Fixed Timer"])
video_file = st.sidebar.file_uploader("Upload traffic video", type=["mp4", "avi"])
audio_file = st.sidebar.file_uploader("Upload siren audio (optional)", type=["wav", "flac", "mp3"])
min_green = st.sidebar.slider("Min Green (s)", 3, 8, 5)
max_green = st.sidebar.slider("Max Green (s)", 10, 30, 20)
service_rate = st.sidebar.slider("Vehicles/second (Green)", 1, 5, 2)
emission_rate = st.sidebar.slider("Idle emission (g CO₂/vehicle/s)", 0.01, 0.3, 0.06, 0.01)
run_time = st.sidebar.slider("Simulation Duration (s)", 10, 120, 60)
start_btn = st.sidebar.button("▶ Start Simulation")

# placeholders
col1, col2 = st.columns([2, 1])
video_ph = col1.empty()
sim_ph = col2.empty()
charts_ph = st.empty()
log_ph = st.empty()

def draw_intersection(controller, mode):
    sim = np.ones((360, 360, 3), dtype=np.uint8) * 230
    center = (180, 180)
    road = 110
    cv2.rectangle(sim, (center[0] - road // 2, 0), (center[0] + road // 2, 360), (100, 100, 100), -1)
    cv2.rectangle(sim, (0, center[1] - road // 2), (360, center[1] + road // 2), (100, 100, 100), -1)
    def color(s): return (0, 200, 0) if s == "GREEN" else (200, 0, 0)
    offsets = {
        "N": (center[0] - 10, center[1] - road // 2 + 8),
        "S": (center[0] + 10, center[1] + road // 2 - 30),
        "W": (center[0] - road // 2 + 8, center[1] + 10),
        "E": (center[0] + road // 2 - 30, center[1] - 10),
    }
    for l, pos in offsets.items():
        cv2.rectangle(sim, (pos[0], pos[1]), (pos[0] + 20, pos[1] + 20), color(controller.state[l]), -1)
        cv2.putText(sim, f"{l}:{controller.queue[l]}", (pos[0] - 10, pos[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(sim, f"Mode: {mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return cv2.cvtColor(sim, cv2.COLOR_BGR2RGB)

# Only instantiate detector when user presses Start to avoid heavy init on load
if start_btn:
    st.sidebar.success("Starting simulation...")
    detector = LaneDetector()  # lazy-safe loader inside class
    if controller_type == "Adaptive AI":
        controller = FourWayController(min_green=min_green, max_green=max_green,
                                       service_rate=service_rate, emission_rate=emission_rate)
    else:
        controller = FixedController(green_time=(min_green + max_green) // 2,
                                     service_rate=service_rate, emission_rate=emission_rate)

    # prepare video source
    if video_file:
        try:
            tmp_path = "temp_video.mp4"
            with open(tmp_path, "wb") as f:
                f.write(video_file.read())
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                cap = None
        except Exception as e:
            st.sidebar.error("Failed to open uploaded video, using blank frames.")
            cap = None
    else:
        cap = None

    start_time = time.time()
    # run simulation loop for run_time seconds
    while time.time() - start_time < run_time:
        try:
            if cap:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            counts, amb_lane, annotated = detector.predict_frame(frame)

            siren_lane = None
            if audio_file is not None:
                try:
                    audio_bytes = audio_file.read()
                    detected, score = detect_siren_from_audio_bytes(audio_bytes)
                    if detected:
                        # heuristics: prioritize lane with highest count if siren present
                        siren_lane = max(counts, key=lambda x: counts[x])
                except Exception as e:
                    # ignore audio errors
                    siren_lane = None

            rec = controller.step(counts, ambulance_lane=(amb_lane or siren_lane), dt=1.0)

            # prepare annotated image (ensure numpy)
            if isinstance(annotated, np.ndarray):
                overlay = annotated.copy()
            else:
                overlay = frame.copy()

            info = f"N:{counts['N']} E:{counts['E']} S:{counts['S']} W:{counts['W']} | Green:{rec['green_lane']} | Mode:{rec['mode']}"
            cv2.putText(overlay, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            video_ph.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")

            sim_ph.image(draw_intersection(controller, rec["mode"]))
            df = pd.DataFrame(controller.history)
            if not df.empty:
                charts_ph.line_chart(df[["total_waiting", "emissions_g"]].assign(index=df.index).set_index("index"))
                log_ph.dataframe(df.tail(6))
            time.sleep(1.0)

        except Exception as e:
            # never crash the app - show error and continue
            st.sidebar.warning(f"Runtime warning (continuing): {e}")
            time.sleep(1.0)
            continue

    # export results
    out_fn = f"results_{controller_type.lower().replace(' ', '_')}.csv"
    controller.export_log(out_fn)
    st.success("Simulation complete — results saved.")

    # evaluation
    if os.path.exists(out_fn):
        res_df = pd.read_csv(out_fn)
        avg_queue = res_df["total_waiting"].mean()
        tot_em = res_df["emissions_g"].sum()
        st.subheader("Performance Evaluation")
        st.metric("Average Queue", f"{avg_queue:.2f}")
        st.metric("Total Emissions (g CO₂)", f"{tot_em:.2f}")

    # comparison if both exist
    if os.path.exists("results_adaptive_ai.csv") and os.path.exists("results_fixed_timer.csv"):
        st.markdown("### 🔬 Adaptive vs Fixed — Comparison")
        a = pd.read_csv("results_adaptive_ai.csv")
        f = pd.read_csv("results_fixed_timer.csv")
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].bar(["Adaptive", "Fixed"], [a["total_waiting"].mean(), f["total_waiting"].mean()], color=["green", "red"])
        ax[0].set_title("Average Queue")
        ax[1].bar(["Adaptive", "Fixed"], [a["emissions_g"].sum(), f["emissions_g"].sum()], color=["green", "red"])
        ax[1].set_title("Total Emissions")
        st.pyplot(fig)

    st.download_button("Download this run CSV", pd.read_csv(out_fn).to_csv(index=False), out_fn, "text/csv")

else:
    # When not running, show helpful instructions so app remains alive
    st.info("Upload a short traffic video (mp4/avi) and optionally a siren audio, adjust sliders, then click ▶ Start Simulation.")
    st.markdown("""
    **Notes**
    - The app will lazy-load the YOLOv8n model when you press Start.
    - If YOLO / PyTorch cannot initialize on your machine, the app **automatically falls back to a mock detector** so you can still demo the controllers, visualization and evaluation.
    - Keep uploaded videos short (<= 60s) and 720p or lower for responsiveness.
    """)
