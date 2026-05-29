# face-attendance-system
# AI-Powered Facial Recognition Attendance System

An intelligent attendance management system built using state-of-the-art 
face detection and recognition models, developed as a capstone project 
during internship at Krugna Technologies, Bengaluru.

---

## Overview

Traditional attendance systems are slow and prone to proxy attendance. 
This system automates the process using real-time facial recognition — 
detecting, verifying, and marking attendance with high accuracy and 
anti-spoofing protection.

---

## Tech Stack

| Component | Technology |
|---|---|
| Face Detection | RetinaFace |
| Face Recognition | ArcFace (512-dim embeddings) |
| Similarity Search | FAISS (Facebook AI Similarity Search) |
| Anti-Spoofing | MobileNetV2 |
| Frontend/UI | Streamlit |
| Language | Python |

---

##  Features

-  Real-time face detection using RetinaFace
-  Anti-spoofing to prevent photo/video attacks
-  High-accuracy recognition using ArcFace embeddings
-  Fast similarity search with FAISS
-  Streamlit dashboard for data collection and attendance view
-  Multi-face detection in a single frame

---

##  Project Structure
face-attendance-system/
│
├── data_collection/        # Streamlit app for registering new faces
├── recognition/            # ArcFace + FAISS pipeline
├── detection/              # RetinaFace detection module
├── antispoofing/           # MobileNetV2 liveness detection
├── attendance/             # Attendance logging and export
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
└── README.md

---

## 🏗️ System Architecture
