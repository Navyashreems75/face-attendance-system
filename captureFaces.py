import streamlit as st
import cv2
import os
import time
import numpy as np
from datetime import datetime

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Data Collector",
    page_icon="📸",
    layout="centered"
)

# FIX: CSS was missing — all the custom HTML classes were unstyled.
# Added the full stylesheet that the st.markdown(...unsafe_allow_html=True) calls reference.
st.markdown("""
<style>
/* ── General ── */
.main-header { text-align: center; padding: 1rem 0 0.5rem; }
.main-header h1 { font-size: 2rem; font-weight: 700; color: #00D4C8; margin-bottom: 0.2rem; }
.main-header p  { color: #94A3B8; font-size: 0.9rem; }

/* ── Status banners ── */
.success-banner {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 10px;
    padding: 14px 18px;
    color: #d1fae5;
    font-size: 0.92rem;
    margin: 10px 0;
}
.warning-banner {
    background: linear-gradient(135deg, #451a03, #78350f);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    color: #fef3c7;
    font-size: 0.92rem;
    margin: 10px 0;
}

/* ── Info card ── */
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
}

/* ── Stat row ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin: 10px 0;
}
.stat-box {
    flex: 1;
    background: rgba(0,212,200,0.08);
    border: 1px solid rgba(0,212,200,0.2);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.stat-box .num   { font-size: 2rem; font-weight: 700; color: #00D4C8; font-family: monospace; }
.stat-box .label { font-size: 0.75rem; color: #64748B; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Progress bar ── */
.progress-bar-bg {
    height: 8px;
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #00D4C8, #0ea5e9);
    border-radius: 4px;
    transition: width 0.3s ease;
}

/* ── Tip box ── */
.tip-box {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #00D4C8;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 0.85rem;
    color: #94A3B8;
    margin-bottom: 4px;
}

/* ── Student chip ── */
.student-chip {
    display: inline-block;
    background: rgba(0,212,200,0.12);
    border: 1px solid rgba(0,212,200,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #00D4C8;
    margin-left: 6px;
}
.student-chip.done {
    background: rgba(16,185,129,0.12);
    border-color: rgba(16,185,129,0.25);
    color: #10b981;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📸 Face Data Collector</h1>
    <p>Capture student photos for attendance system training</p>
</div>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────
if "captured_count" not in st.session_state:
    st.session_state.captured_count = 0
if "capturing" not in st.session_state:
    st.session_state.capturing = False
if "registered_students" not in st.session_state:
    st.session_state.registered_students = []
if "last_capture_time" not in st.session_state:
    st.session_state.last_capture_time = 0

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    photos_target = st.slider(
        "Photos per student",
        min_value=5, max_value=20, value=10,
        help="More photos = better accuracy"
    )

    capture_delay = st.slider(
        "Delay between captures (sec)",
        min_value=0.5, max_value=3.0, value=1.0, step=0.5
    )

    save_folder = st.text_input(
        "Save folder",
        value="faces",
        help="Root folder where student photos are saved"
    )

    st.markdown("---")
    st.markdown("### 📋 Photo Tips")
    tips = [
        "😐 Neutral expression",
        "🙂 Slight smile",
        "↙️ Face slightly left",
        "↗️ Face slightly right",
        "💡 Near window light",
        "🏠 Indoor light",
        "👓 With glasses",
        "🚫 Without glasses",
        "⬆️ Look slightly up",
        "⬇️ Look slightly down",
    ]
    for tip in tips[:photos_target]:
        st.markdown(f"<div class='tip-box'>{tip}</div>", unsafe_allow_html=True)

# ─── Student Info Form ─────────────────────────────────────────────────────────
st.markdown("### 👤 Student Details")

col1, col2 = st.columns(2)
with col1:
    student_name = st.text_input("Full Name", placeholder="e.g. Arjun Sharma")
with col2:
    roll_number = st.text_input("Roll Number", placeholder="e.g. 101")

col3, col4 = st.columns(2)
with col3:
    branch = st.selectbox("Branch", ["CSE", "ISE", "ECE", "EEE", "MECH", "CIVIL", "MBA"])
with col4:
    semester = st.selectbox("Semester", ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"])

# ─── Stats Row ─────────────────────────────────────────────────────────────────
total_students = len(st.session_state.registered_students)
total_photos = sum(s["photos"] for s in st.session_state.registered_students) if st.session_state.registered_students else 0

st.markdown(f"""
<div class="stat-row">
    <div class="stat-box">
        <div class="num">{total_students}</div>
        <div class="label">Students Registered</div>
    </div>
    <div class="stat-box">
        <div class="num">{total_photos}</div>
        <div class="label">Total Photos</div>
    </div>
    <div class="stat-box">
        <div class="num">{st.session_state.captured_count}</div>
        <div class="label">Current Session</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Capture Section ───────────────────────────────────────────────────────────
st.markdown("### 🎥 Capture Photos")

ready = student_name.strip() and roll_number.strip()

if not ready:
    st.markdown("<div class='warning-banner'>⚠️ Please enter student name and roll number first</div>", unsafe_allow_html=True)
else:
    folder_name = f"{student_name.strip().replace(' ', '_')}_{roll_number.strip()}"
    save_path = os.path.join(save_folder, folder_name)

    st.markdown(f"""
    <div class="info-card">
        <b style="color:#00D4C8">Save Path:</b>
        <span style="font-family:'JetBrains Mono',monospace; color:#94A3B8; font-size:0.9rem">
            {save_path}/
        </span>
    </div>
    """, unsafe_allow_html=True)

    progress_pct = int((st.session_state.captured_count / photos_target) * 100)
    st.markdown(f"""
    <div style="margin: 0.5rem 0;">
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#64748B; margin-bottom:4px;">
            <span>Progress</span>
            <span style="font-family:'JetBrains Mono',monospace; color:#00D4C8">
                {st.session_state.captured_count}/{photos_target}
            </span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width:{min(progress_pct, 100)}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    camera_placeholder = st.empty()
    status_placeholder = st.empty()

    col_start, col_stop = st.columns(2)

    with col_start:
        start_btn = st.button("▶ Start Capturing", key="start", disabled=st.session_state.capturing)
    with col_stop:
        stop_btn = st.button("⏹ Stop", key="stop")

    if stop_btn:
        st.session_state.capturing = False

    if start_btn and ready:
        os.makedirs(save_path, exist_ok=True)
        st.session_state.capturing = True
        st.session_state.captured_count = 0

        # FIX: use InsightFace for face detection during capture to match
        # the detector used in generateEmbeddings.py (both now use RetinaFace).
        # Old code used Haar cascade which is less accurate and inconsistent
        # with the ArcFace embedding pipeline.
        from insightface.app import FaceAnalysis as _FaceAnalysis
        _capture_app = _FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        _capture_app.prepare(ctx_id=0, det_size=(640, 640))

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.markdown("<div class='warning-banner'>❌ Cannot open webcam. Please check your camera.</div>", unsafe_allow_html=True)
            st.session_state.capturing = False
        else:
            captured   = 0
            last_capture = 0

            while st.session_state.capturing and captured < photos_target:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()
                display_frame = frame.copy()

                # Run InsightFace detection
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces   = _capture_app.get(img_rgb)
                face_detected = len(faces) > 0

                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    color = (0, 212, 200)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, "Face Detected", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.putText(display_frame, f"Captured: {captured}/{photos_target}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 200), 2)
                cv2.putText(display_frame, f"Student: {student_name}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if face_detected and (current_time - last_capture) >= capture_delay:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    img_path = os.path.join(save_path, f"{captured+1}_{timestamp}.jpg")
                    cv2.imwrite(img_path, frame)
                    captured += 1
                    last_capture = current_time
                    st.session_state.captured_count = captured

                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]),
                                 (0, 212, 200), -1)
                    display_frame = cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0)

                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                if face_detected:
                    status_placeholder.markdown(f"""
                    <div class='success-banner'>
                        ✅ Face detected — Auto capturing... {captured}/{photos_target}
                    </div>""", unsafe_allow_html=True)
                else:
                    status_placeholder.markdown("""
                    <div class='warning-banner'>
                        👀 No face detected — Please look at the camera
                    </div>""", unsafe_allow_html=True)

            cap.release()
            st.session_state.capturing = False

            if captured >= photos_target:
                st.session_state.registered_students.append({
                    "name": student_name,
                    "roll": roll_number,
                    "branch": branch,
                    "sem": semester,
                    "photos": captured,
                    "folder": save_path,
                    "time": datetime.now().strftime("%H:%M:%S")
                })

                camera_placeholder.empty()
                status_placeholder.empty()
                st.markdown(f"""
                <div class='success-banner'>
                    🎉 Done! {captured} photos saved for {student_name} ({roll_number})<br>
                    <span style='font-size:0.85rem; opacity:0.8'>Saved to: {save_path}/</span>
                </div>""", unsafe_allow_html=True)
                st.balloons()

# ─── Registered Students List ──────────────────────────────────────────────────
if st.session_state.registered_students:
    st.markdown("---")
    st.markdown("### ✅ Registered Students")

    for s in st.session_state.registered_students:
        st.markdown(f"""
        <div class="info-card" style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="color:#E2E8F0; font-weight:600">{s['name']}</span>
                <span class="student-chip done">{s['roll']}</span>
                <span class="student-chip">{s['branch']} • {s['sem']} Sem</span>
            </div>
            <div style="text-align:right">
                <span style="color:#00D4C8; font-family:'JetBrains Mono',monospace; font-weight:600">{s['photos']} 📸</span>
                <br><span style="color:#475569; font-size:0.75rem">{s['time']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📤 Summary")

    summary_lines = ["Name,Roll,Branch,Semester,Photos,Folder"]
    for s in st.session_state.registered_students:
        summary_lines.append(f"{s['name']},{s['roll']},{s['branch']},{s['sem']},{s['photos']},{s['folder']}")
    summary_csv = "\n".join(summary_lines)

    st.download_button(
        label="⬇️ Download Registration Summary (CSV)",
        data=summary_csv,
        file_name=f"registration_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ─── Instructions ──────────────────────────────────────────────────────────────
with st.expander("📖 How to Run This App"):
    st.markdown("""
    **Step 1 — Activate your conda environment:**
    ```bash
    conda activate face_attendance
    ```

    **Step 2 — Run the app:**
    ```bash
    streamlit run captureFaces.py
    ```

    **Step 3 — For each student:**
    1. Enter their Name and Roll Number
    2. Select Branch and Semester
    3. Click **Start Capturing**
    4. Student looks at camera — photos auto-save when face detected
    5. Done when progress bar reaches 100%

    **Step 4 — After all students enrolled:**
    ```bash
    python generateEmbeddings.py
    python buildFaissIndex.py
    ```
    """)
