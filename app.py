"""
app.py
──────
Streamlit dashboard for the AI Traffic Intelligence system.
"""

import streamlit as st
import pandas as pd
import os
from processor import process_video

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Traffic Intelligence Dashboard",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 AI-Powered Traffic Queue Analysis & Rule Violation Detection")
st.caption("Upload a traffic video → YOLO + Deep SORT → real-time analytics")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

confidence_filter = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05)
frame_skip = st.sidebar.slider("Frame Skip (1 = every frame)", 1, 5, 2)
junction = st.sidebar.text_input("Junction Name", "Main City Signal")
lanes = st.sidebar.number_input("Number of Lanes", 1, 8, 4)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Traffic Light ROI** — leave at 0 to auto‑detect from the top‑right "
    "of the frame, or enter pixel coordinates for precision."
)
roi_x1 = st.sidebar.number_input("ROI x1", 0, 1920, 0)
roi_y1 = st.sidebar.number_input("ROI y1", 0, 1080, 0)
roi_x2 = st.sidebar.number_input("ROI x2", 0, 1920, 0)
roi_y2 = st.sidebar.number_input("ROI y2", 0, 1080, 0)

traffic_light_roi = None
if roi_x2 > roi_x1 and roi_y2 > roi_y1:
    traffic_light_roi = (roi_x1, roi_y1, roi_x2, roi_y2)

# ── File Upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

if uploaded_file is None:
    st.info("👆 Upload a traffic video to begin analysis.")
    st.stop()

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

input_path = os.path.join("uploads", uploaded_file.name)
output_video = "outputs/yolo_output.mp4"
output_csv = "outputs/traffic_data.csv"

# Save uploaded file
with open(input_path, "wb") as f:
    f.write(uploaded_file.read())

#── Processing ───────────────────────────────────────────────────────────────
if "processed" not in st.session_state or st.sidebar.button("🔄 Re‑process Video"):
    progress_bar = st.progress(0.0)
    with st.spinner("Running YOLO + Deep SORT pipeline..."):
        process_video(
            input_path, output_video, output_csv,
            traffic_light_roi=traffic_light_roi,
            frame_skip=frame_skip,
            progress_callback=lambda p: progress_bar.progress(p),
        )
    progress_bar.empty()
    st.session_state.processed = True
    st.success("✅ Processing complete!")

# ── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv(output_csv)

if df.empty:
    st.warning("No vehicles detected in this video. Try a different clip.")
    st.stop()

# Apply confidence filter
df = df[df["confidence"] >= confidence_filter]

# ── KPIs ─────────────────────────────────────────────────────────────────────
st.markdown("---")
k1, k2, k3, k4, k5 = st.columns(5)

total_vehicles = df["vehicle_id"].nunique()
queued = df[df["in_queue"]]["vehicle_id"].nunique()
violations = df[df["red_light_violation"]]["vehicle_id"].nunique()
rash = df[df["rash_driving"]]["vehicle_id"].nunique()
total_frames = int(df["frame"].max()) if not df.empty else 0

k1.metric("🚗 Vehicles Tracked", total_vehicles)
k2.metric("📦 Queued Vehicles", queued)
k3.metric("🚨 Red‑Light Violations", violations)
k4.metric("⚠️ Rash Driving", rash)
k5.metric("🎞️ Frames Processed", total_frames)

st.markdown("---")

# ── Video Comparison ─────────────────────────────────────────────────────────
st.subheader("🎥 Video Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**📥 Raw Input**")
    st.video(input_path)
with col2:
    st.markdown("**🤖 YOLO Annotated Output**")
    if os.path.exists(output_video):
        st.video(output_video)
    else:
        st.warning("Output video not found.")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 Queue Analysis",
    "🚨 Violations",
    "📊 Traffic Flow",
    "🚗 Vehicle Types",
    "📁 Reports",
])

# ── Queue Tab ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Queue Analysis")

    queue_over_time = (
        df[df["in_queue"]]
        .groupby("frame")["vehicle_id"]
        .nunique()
        .reset_index()
    )
    queue_over_time.columns = ["Frame", "Queued Vehicles"]

    if not queue_over_time.empty:
        st.line_chart(queue_over_time.set_index("Frame"))
    else:
        st.info("No queue events detected.")

    st.metric("Peak Queue Length", int(queue_over_time["Queued Vehicles"].max()) if not queue_over_time.empty else 0)

# ── Violations Tab ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Violation Records")

    violation_df = df[df["red_light_violation"] | df["rash_driving"]]

    if not violation_df.empty:
        st.dataframe(
            violation_df[[
                "frame", "vehicle_id", "vehicle_type", "confidence",
                "speed_px", "red_light_violation", "rash_driving", "signal_state",
            ]],
            use_container_width=True,
        )

        st.markdown("**Violations over time**")
        viol_time = (
            violation_df.groupby("frame")["vehicle_id"]
            .nunique()
            .reset_index()
        )
        viol_time.columns = ["Frame", "Violating Vehicles"]
        st.line_chart(viol_time.set_index("Frame"))
    else:
        st.success("No violations detected!")

# ── Traffic Flow Tab ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Traffic Flow Over Time")
    flow = df.groupby("frame")["vehicle_id"].nunique().reset_index()
    flow.columns = ["Frame", "Active Vehicles"]
    st.line_chart(flow.set_index("Frame"))

    st.subheader("Average Speed Over Time")
    speed_flow = df.groupby("frame")["speed_px"].mean().reset_index()
    speed_flow.columns = ["Frame", "Avg Speed (px/frame)"]
    st.line_chart(speed_flow.set_index("Frame"))

# ── Vehicle Types Tab ────────────────────────────────────────────────────────
with tab4:
    st.subheader("Vehicle Type Distribution")
    st.bar_chart(df.drop_duplicates("vehicle_id")["vehicle_type"].value_counts())

# ── Reports Tab ──────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Download Reports")

    st.download_button(
        "📄 Full Detection CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"traffic_report_{junction.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # Summary report
    summary = {
        "Junction": junction,
        "Total Vehicles": total_vehicles,
        "Queued Vehicles": queued,
        "Red-Light Violations": violations,
        "Rash Driving Incidents": rash,
        "Frames Processed": total_frames,
    }
    summary_df = pd.DataFrame([summary])
    st.download_button(
        "📊 Summary Report CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"summary_{junction.replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("AI Traffic Intelligence | YOLOv8 + Deep SORT + HSV Signal Detection + Streamlit")
