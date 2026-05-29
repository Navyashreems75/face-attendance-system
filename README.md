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
 data_collection/        # Streamlit app for registering new faces.
├── recognition/            # ArcFace + FAISS pipeline
├── detection/              # RetinaFace detection module
├── antispoofing/           # MobileNetV2 liveness detection
├── attendance/             # Attendance logging and export
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
└── README.md

---
Output Images
## Register Page
<img width="1428" height="838" alt="register page" src="https://github.com/user-attachments/assets/8cb4a1f8-c0ab-4f1e-8c17-6ada76a20c9f" />

## Login Page
<img width="1761" height="855" alt="login page" src="https://github.com/user-attachments/assets/caffde28-9e58-4f16-9bf2-74bfb6fbe16a" />

## Admin Dashboard
<img width="1886" height="892" alt="admin dashboard" src="https://github.com/user-attachments/assets/8649bcb9-709a-4391-8689-f6413efea060" />

## Teacher Dashboard
<img width="1879" height="895" alt="teacher dashboard" src="https://github.com/user-attachments/assets/87bff302-662c-4b24-91b6-f2e60548b1f4" />

## Kisok Page
<img width="1876" height="880" alt="kiosk page" src="https://github.com/user-attachments/assets/1bcdd128-dbb0-47e3-a16b-1e939338fd5e" />

## New Enrollement Page
<img width="1886" height="884" alt="new enrollment page" src="https://github.com/user-attachments/assets/0a523a54-b689-4d52-9c05-0b5409b7a761" />

## Student Dashboard 
<img width="671" height="886" alt="student dashboard" src="https://github.com/user-attachments/assets/f53ac3bb-c945-4eaa-88e6-2eef7590dc91" />

## live Face Recognition
<img width="1862" height="890" alt="Screenshot 2026-05-03 215248" src="https://github.com/user-attachments/assets/f0d7c77f-d2b5-40b0-88cb-c19e29ddf2e0" />
