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

# ── Processing ───────────────────────────────────────────────────────────────
# ── Processing ───────────────────────────────────────────────────────────────
if "processed" not in st.session_state or st.sidebar.button("🔄 Re‑process Video"):
    with st.spinner("Running YOLO + Deep SORT pipeline..."):
        try:
            process_video(
                input_path,
                output_video,
                output_csv,
                traffic_light_roi=traffic_light_roi,
                frame_skip=frame_skip,
            )
        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            st.stop()
    
    # ← ADD THIS DEBUG BLOCK
    if os.path.exists(output_video):
        st.success("✅ Processing complete!")
        st.session_state.processed = True
    else:
        st.error(f"⚠️ Output video not created. Check terminal for errors.")
        st.write(f"Expected at: {output_video}")
        st.stop()essing complete!")

# ── Evaluation Section ──────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_tracking(df):
    """Analyze tracking quality"""
    track_lengths = df.groupby('vehicle_id').size()
    return {
        'avg_track_length': track_lengths.mean(),
        'max_track_length': track_lengths.max(),
        'min_track_length': track_lengths.min(),
        'short_tracks': (track_lengths < 5).sum(),
        'long_tracks': (track_lengths > 50).sum()
    }

def iou(box1, box2):
    """Calculate Intersection over Union"""
    intersection = (
        max(0, min(box1['x2'], box2['x2']) - max(box1['x1'], box2['x1'])) *
        max(0, min(box1['y2'], box2['y2']) - max(box1['y1'], box2['y1']))
    )
    union = (
        (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1']) +
        (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1']) - intersection
    )
    return intersection / union if union > 0 else 0

def evaluate_with_ground_truth(pred_df, gt_df):
    """Compare predictions with annotated ground truth"""
    tp, fp, fn = 0, 0, 0

    for frame_id in pred_df['frame_id'].unique():
        pred_frame = pred_df[pred_df['frame_id'] == frame_id]
        gt_frame = gt_df[gt_df['frame_id'] == frame_id]

        matched = 0
        for _, pred in pred_frame.iterrows():
            for _, gt in gt_frame.iterrows():
                if iou(pred, gt) > 0.5:
                    matched += 1
                    tp += 1
                    break

        fp += len(pred_frame) - matched
        fn += max(0, len(gt_frame) - matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ── Streamlit Evaluation Dashboard ──────────────────────────────────────────

st.markdown("---")
st.header("📊 Project Evaluation Dashboard")

# Load detection results
df = pd.read_csv(output_csv)

if df.empty:
    st.warning("No detections found. Cannot evaluate.")
    st.stop()

# ── Tab Layout ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Basic Stats",
    "🎯 Confidence",
    "🔄 Tracking Quality",
    "🚦 Traffic Light",
    "✔️ Ground Truth"
])

# ── TAB 1: Basic Statistics ─────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Detection Statistics")

    total_frames = df['frame_id'].max() + 1
    total_detections = len(df)
    unique_vehicles = df['vehicle_id'].nunique()
    avg_per_frame = total_detections / total_frames

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Frames", f"{total_frames:,}")
    col2.metric("Total Detections", f"{total_detections:,}")
    col3.metric("Unique Vehicles", f"{unique_vehicles:,}")
    col4.metric("Avg Detections/Frame", f"{avg_per_frame:.2f}")

    # Frame-by-frame analysis
    st.subheader("📹 Frame Analysis")
    frame_counts = df.groupby('frame_id').size()
    frames_with_det = len(frame_counts)
    frames_without_det = total_frames - frames_with_det

    col1, col2, col3 = st.columns(3)
    col1.metric("Frames with Detections", frames_with_det)
    col2.metric("Frames without Detections", frames_without_det)
    col3.metric("Max Detections in 1 Frame", frame_counts.max())

    # Detections over time chart
    st.subheader("📈 Detections Over Time")
    st.line_chart(frame_counts, use_container_width=True)

    # Vehicle type distribution
    if 'class_name' in df.columns:
        st.subheader("🚗 Vehicle Type Distribution")
        vehicle_counts = df['class_name'].value_counts()
        st.bar_chart(vehicle_counts, use_container_width=True)

# ── TAB 2: Confidence Analysis ──────────────────────────────────────────────
with tab2:
    st.subheader("🎯 Confidence Score Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Confidence", f"{df['confidence'].mean():.4f}")
    col2.metric("Min Confidence", f"{df['confidence'].min():.4f}")
    col3.metric("Max Confidence", f"{df['confidence'].max():.4f}")

    col1, col2 = st.columns(2)
    col1.metric("Detections ≥ 0.5", f"{(df['confidence'] >= 0.5).sum():,}")
    col2.metric("Detections ≥ 0.8", f"{(df['confidence'] >= 0.8).sum():,}")

    # Confidence distribution histogram
    st.subheader("📊 Confidence Distribution")
    hist_data = pd.cut(df['confidence'], bins=10).value_counts().sort_index()
    st.bar_chart(hist_data, use_container_width=True)

    # Quality rating
    mean_conf = df['confidence'].mean()
    if mean_conf >= 0.80:
        st.success(f"🟢 Excellent! Mean confidence {mean_conf:.2f} is above 0.80")
    elif mean_conf >= 0.70:
        st.info(f"🟡 Good. Mean confidence {mean_conf:.2f} is above 0.70")
    else:
        st.warning(f"🔴 Needs improvement. Mean confidence {mean_conf:.2f} is below 0.70")

# ── TAB 3: Tracking Quality ─────────────────────────────────────────────────
with tab3:
    st.subheader("🔄 Deep SORT Tracking Quality")

    tracking = analyze_tracking(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Track Length", f"{tracking['avg_track_length']:.1f} frames")
    col2.metric("Longest Track", f"{tracking['max_track_length']} frames")
    col3.metric("Shortest Track", f"{tracking['min_track_length']} frames")

    col1, col2 = st.columns(2)
    col1.metric("Short Tracks (<5 frames)", tracking['short_tracks'],
                help="High number = IDs switching frequently")
    col2.metric("Long Tracks (>50 frames)", tracking['long_tracks'],
                help="High number = stable tracking")

    # Track length distribution
    st.subheader("📊 Track Length Distribution")
    track_lengths = df.groupby('vehicle_id').size().reset_index(name='length')
    st.bar_chart(track_lengths.set_index('vehicle_id')['length'], use_container_width=True)

    # Quality rating
    short_ratio = tracking['short_tracks'] / df['vehicle_id'].nunique()
    if short_ratio < 0.1:
        st.success(f"🟢 Excellent! Only {short_ratio:.0%} short tracks. Tracking is very stable.")
    elif short_ratio < 0.3:
        st.info(f"🟡 Good. {short_ratio:.0%} short tracks. Tracking is fairly stable.")
    else:
        st.warning(f"🔴 Needs improvement. {short_ratio:.0%} short tracks. IDs may be switching too often.")

# ── TAB 4: Traffic Light ────────────────────────────────────────────────────
with tab4:
    st.subheader("🚦 Traffic Light Detection")

    if 'traffic_light_state' in df.columns:
        state_counts = df['traffic_light_state'].value_counts()

        cols = st.columns(len(state_counts))
        for i, (state, count) in enumerate(state_counts.items()):
            cols[i].metric(f"{state}", f"{count:,}")

        st.bar_chart(state_counts, use_container_width=True)
    else:
        st.info("No traffic light state column found in CSV. "
                "Make sure your pipeline outputs 'traffic_light_state' in the CSV.")

# ── TAB 5: Ground Truth Comparison ──────────────────────────────────────────
with tab5:
    st.subheader("✔️ Ground Truth Comparison")
    st.markdown("""
    Upload a **ground truth CSV** with the same format as your output CSV 
    (columns: `frame_id`, `x1`, `y1`, `x2`, `y2`) to compare accuracy.
    """)

    gt_file = st.file_uploader("Upload Ground Truth CSV", type=["csv"])

    if gt_file is not None:
        gt_df = pd.read_csv(gt_file)
        st.write("Ground Truth Preview:")
        st.dataframe(gt_df.head(10))

        if st.button("🔍 Evaluate Accuracy"):
            with st.spinner("Calculating metrics..."):
                results = evaluate_with_ground_truth(df, gt_df)

            st.markdown("### 📊 Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{results['precision']:.2%}")
            col2.metric("Recall", f"{results['recall']:.2%}")
            col3.metric("F1-Score", f"{results['f1']:.2%}")

            col1, col2, col3 = st.columns(3)
            col1.metric("True Positives", results['tp'])
            col2.metric("False Positives", results['fp'])
            col3.metric("False Negatives", results['fn'])

            # Quality rating
            if results['f1'] >= 0.85:
                st.success("🟢 Excellent! F1-Score is above 85%")
            elif results['f1'] >= 0.70:
                st.info("🟡 Good. F1-Score is above 70%")
            else:
                st.warning("🔴 Needs improvement. F1-Score is below 70%")

            # Detailed breakdown
            st.markdown("### 📋 What These Metrics Mean")
            st.markdown(f"""
            | Metric | Value | Meaning |
            |--------|-------|---------|
            | **Precision** | {results['precision']:.2%} | Of all detections, how many were correct |
            | **Recall** | {results['recall']:.2%} | Of all actual vehicles, how many were found |
            | **F1-Score** | {results['f1']:.2%} | Balance between Precision and Recall |
            """)
    else:
        st.info("💡 **No ground truth?** You can still evaluate using the other tabs. "
                "For ground truth, manually annotate 50-100 frames using tools like "
                "[CVAT](https://cvat.ai) or [LabelImg](https://github.com/heartexlabs/labelImg).")


# ── Overall Summary ─────────────────────────────────────────────────────────
st.markdown("---")
st.header("🏆 Overall Project Health")

mean_conf = df['confidence'].mean()
tracking = analyze_tracking(df)
short_ratio = tracking['short_tracks'] / df['vehicle_id'].nunique()

scores = []

# Detection score
if mean_conf >= 0.80:
    scores.append(("Detection Quality", "🟢 Excellent", mean_conf))
elif mean_conf >= 0.70:
    scores.append(("Detection Quality", "🟡 Good", mean_conf))
else:
    scores.append(("Detection Quality", "🔴 Needs Work", mean_conf))

# Tracking score
if short_ratio < 0.1:
    scores.append(("Tracking Stability", "🟢 Excellent", 1 - short_ratio))
elif short_ratio < 0.3:
    scores.append(("Tracking Stability", "🟡 Good", 1 - short_ratio))
else:
    scores.append(("Tracking Stability", "🔴 Needs Work", 1 - short_ratio))

# Coverage score
coverage = len(df.groupby('frame_id').size()) / (df['frame_id'].max() + 1)
if coverage >= 0.9:
    scores.append(("Frame Coverage", "🟢 Excellent", coverage))
elif coverage >= 0.7:
    scores.append(("Frame Coverage", "🟡 Good", coverage))
else:
    scores.append(("Frame Coverage", "🔴 Needs Work", coverage))

# Display
cols = st.columns(3)
for i, (name, status, score) in enumerate(scores):
    cols[i].metric(name, status, f"{score:.2%}")

# Overall score
overall = np.mean([s[2] for s in scores])
st.progress(overall)
st.markdown(f"### Overall Score: **{overall:.0%}**")

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
