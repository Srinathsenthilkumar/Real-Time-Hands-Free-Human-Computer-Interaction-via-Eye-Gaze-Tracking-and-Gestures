# Real-Time-Hands-Free-Human-Computer-Interaction-via-Eye-Gaze-Tracking-and-Gestures
Real-time hands-free HCI system using MediaPipe Face Mesh, OpenCV &amp; PyAutoGUI. Controls cursor movement via eye-gaze tracking and executes left/right clicks and scrolling using facial micro-gestures (winks &amp; mouth open). Lightweight, CPU-friendly assistive AI designed for motor-impaired accessibility.
# 👁️ Real-Time Hands-Free HCI via Eye-Gaze Tracking & Gestures

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange.svg)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, lightweight Assistive AI system that allows users to control system cursor movements, mouse clicks, and page scrolling purely using **eye movements and facial micro-gestures**. 

Built entirely with standard webcams and **zero heavy GPU requirements**, this project aims to provide motor-impaired individuals with an accessible, low-cost solution for complete hands-free computer operation.

---

## 💡 Key Features

* **🎯 Gaze-Based Cursor Movement:** Maps iris positions to dynamic screen coordinates with low latency.
* **⚡ Jitter Reduction:** Implements Exponential Moving Average (EMA) filtering to smooth out micro-saccadic eye jitter.
* **😉 Gesture-Driven Clicks:** 
  * **Left Eye Wink:** Triggers System Left Click.
  * **Right Eye Wink:** Triggers System Right Click.
* **📜 Mouth Aspect Ratio (MAR) Scrolling:** Open mouth to toggle **Scroll Mode** (move eyes/head up and down to scroll pages).
* **🖥️ Low Hardware Overhead:** Runs at smooth 30–60 FPS on standard consumer CPUs—no dedicated infrared or GPU hardware required.

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python 3.9+
* **Computer Vision:** OpenCV (Video capture & frame processing)
* **Landmark Detection:** MediaPipe Face Mesh (`refine_landmarks=True` for 478 3D facial/iris points)
* **OS Automation:** PyAutoGUI (Hardware event emulation)
* **Math Operations:** NumPy (Euclidean distance, matrix manipulation, EMA filtering)
