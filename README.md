# 🚦 AI-Powered Traffic Queue Analysis & Violation Detection

A computer vision system that analyzes traffic camera footage to detect
vehicles, track them across frames, estimate queue lengths, and flag
rule violations — all displayed through an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## Demo

> 📹 [Watch the demo video on YouTube](YOUR_YOUTUBE_LINK_HERE)

<!-- Add a GIF or screenshot here -->
<!-- ![Dashboard Screenshot](docs/dashboard_screenshot.png) -->

## Features

- **Vehicle Detection** — YOLOv8 detects cars, motorcycles, buses, and
  trucks, filtering out non-vehicle objects.
- **Multi-Object Tracking** — Deep SORT maintains unique IDs across
  frames so each vehicle is counted once.
- **Traffic Signal Detection** — HSV colour thresholding reads the actual
  traffic light state (red / green / yellow) from the video frame.
- **Queue Estimation** — Vehicles that remain stationary for multiple
  consecutive frames are classified as "queued."
- **Red-Light Violation Detection** — Vehicles crossing the stop line
  while the detected signal is red are flagged.
- **Rash Driving Detection** — Smoothed per-vehicle speed estimation
  flags abnormally fast movement.
- **Interactive Dashboard** — Streamlit app with real-time KPIs,
  traffic flow charts, violation logs, and CSV report downloads.

## Architecture
Input Video │ ▼ ┌──────────┐ ┌────────────┐ ┌──────────────┐ │ YOLOv8 │───▶│ Deep SORT │───▶│ Analytics │ │ Detection│ │ Tracking │ │ Engine │ └──────────┘ └────────────┘ └──────┬───────┘ │ ┌────────────────────┼────────────────────┐ │ │ │ ┌─────▼─────┐ ┌────────▼───────┐ ┌───────▼───────┐ │ Traffic │ │ Queue │ │ Violation │ │ Light HSV │ │ Detector │ │ Flags │ └─────┬──────┘ └────────┬───────┘ └───────┬───────┘ │ │ │ └────────────────────┼────────────────────┘ │ ▼ ┌─────────────────┐ │ Streamlit │ │ Dashboard │ └─────────────────┘
