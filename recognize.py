import os
import time
import cv2
import faiss
import pickle
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from antispoofing import AntiSpoofing

# ─── Step 1: Load models ──────────────────────────────────────────────────────
print("Loading ArcFace model...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# SPEED FIX: 320×320 detector is faster on CPU, still accurate at kiosk distance
app.prepare(ctx_id=0, det_size=(320, 320))
print("✅ ArcFace loaded!")

print("Loading Anti-Spoofing model...")
spoof_checker = AntiSpoofing(model_path='antispoofing_best.pth', debug=False)
print("✅ Anti-Spoofing loaded!")

# ─── Step 2: Load FAISS ───────────────────────────────────────────────────────
print("Loading FAISS index...")
index = faiss.read_index("faiss_index.bin")
with open("index_names.pkl", 'rb') as f:
    names = pickle.load(f)
print(f"✅ Loaded {index.ntotal} students!")

# ─── Settings ─────────────────────────────────────────────────────────────────
THRESHOLD = 0.65

# ─── Attendance tracking ──────────────────────────────────────────────────────
attendance_log = {}   # { name: time_str }

def load_existing_attendance():
    """Read today's CSV (if it exists) into attendance_log at startup."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{date_str}.csv"
    if not os.path.exists(filename):
        return
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 3:
            attendance_log[parts[0]] = parts[2]
    if attendance_log:
        print(f"ℹ️  Resuming session — {len(attendance_log)} student(s) already marked today.")

load_existing_attendance()


def mark_attendance(name):
    """Save attendance to CSV. Returns True if newly marked."""
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    filename = f"attendance_{date_str}.csv"

    if name in attendance_log:
        print(f"⚠️  {name} already marked present today at {attendance_log[name]}")
        return False

    file_exists = os.path.exists(filename)
    with open(filename, 'a') as f:
        if not file_exists:
            f.write("Name,Date,Time\n")
        f.write(f"{name},{date_str},{time_str}\n")

    attendance_log[name] = time_str
    print(f"✅ Attendance marked: {name} at {time_str}")
    return True


def student_key_to_display(student_key: str) -> str:
    """
    Convert a student key like "Arjun_Sharma_101" to display name "Arjun Sharma".
    FIX: Old code used split("_")[:-1] which would split "Arjun_Sharma_101"
    into ["Arjun", "Sharma", "101"] and join first two as "Arjun Sharma" —
    correct by coincidence, but fails for single-word names like "Arjun_101"
    (produces empty join). Use rsplit("_", 1) instead which is explicit and safe.
    """
    parts = student_key.rsplit("_", 1)
    return parts[0].replace("_", " ") if len(parts) == 2 else student_key


def draw_hud(frame):
    """Draw the top status bar and last 5 marked students."""
    cv2.putText(frame,
                f"Present today: {len(attendance_log)}  |  Q=quit  R=reset",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    y_offset = 60
    for i, (pname, ptime) in enumerate(list(attendance_log.items())[-5:]):
        display = student_key_to_display(pname)
        cv2.putText(frame, f"OK {display} - {ptime}",
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)


# ─── Non-blocking result display state ───────────────────────────────────────
result_display = {
    'label' : '',
    'color' : (0, 255, 0),
    'until' : 0.0,
    'bbox'  : None,
    'active': False,
}

def set_result(label, color, bbox, duration=2.0):
    result_display['label']  = label
    result_display['color']  = color
    result_display['until']  = time.time() + duration
    result_display['bbox']   = bbox
    result_display['active'] = True


# ─── Open webcam ──────────────────────────────────────────────────────────────
print("\nOpening webcam...")
print("Instructions: Look at camera → Blink once → Hold still for verification")
print("Press Q to quit | Press R to reset")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # ── Result-display cooldown window ────────────────────────────────────────
    if result_display['active']:
        if now < result_display['until']:
            x1, y1, x2, y2 = result_display['bbox']
            color = result_display['color']
            label = result_display['label']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            draw_hud(frame)
            cv2.imshow("Face Attendance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                result_display['active'] = False
                spoof_checker.reset()
                print("🔄 Reset — ready for next person")
            continue
        else:
            result_display['active'] = False
            spoof_checker.reset()

    # ── Run face detection ────────────────────────────────────────────────────
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces   = app.get(img_rgb)

    if faces:
        face            = faces[0]
        bbox            = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        pad_x  = int((x2 - x1) * 0.2)
        pad_y  = int((y2 - y1) * 0.2)
        x1p    = max(0, x1 - pad_x)
        y1p    = max(0, y1 - pad_y)
        x2p    = min(frame.shape[1], x2 + pad_x)
        y2p    = min(frame.shape[0], y2 + pad_y)
        face_crop = frame[y1p:y2p, x1p:x2p]
        if face_crop.size == 0:
            draw_hud(frame)
            cv2.imshow("Face Attendance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                spoof_checker.reset()
            continue

        # ── Stage 1: Blink detection ──────────────────────────────────────────
        blinked, ear, landmarks_ok = spoof_checker.update_blink(frame)

        if not landmarks_ok:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), (128, 128, 128), -1)
            cv2.putText(frame, "Look directly at camera",
                        (x1 + 5, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        elif not blinked:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), (0, 165, 255), -1)
            cv2.putText(frame, f"Please blink to verify (EAR={ear:.2f})",
                        (x1 + 5, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        elif blinked:
            # ── Stage 2: Anti-spoofing ────────────────────────────────────────
            vote_result = spoof_checker.collect_vote(face_crop)
            collected, total = spoof_checker.get_vote_progress()

            if vote_result == 'settling':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 2)
                cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), (200, 200, 0), -1)
                cv2.putText(frame, "Hold still...",
                            (x1 + 5, y1 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif vote_result == 'collecting':
                bar_width = x2 - x1
                filled    = int((collected / total) * bar_width)
                pct       = int((collected / total) * 100)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), (0, 200, 255), -1)
                cv2.putText(frame, f"Verifying... {pct}%",
                            (x1 + 5, y1 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 12), (50, 50, 50), -1)
                cv2.rectangle(frame, (x1, y2 + 2), (x1 + filled, y2 + 12), (0, 200, 255), -1)

            elif vote_result == 'fake':
                set_result("Spoof Detected!", (0, 0, 255), (x1, y1, x2, y2), duration=2.0)

            elif vote_result == 'real':
                # ── Stage 3: Recognition ──────────────────────────────────────
                embedding = face.embedding
                embedding = embedding / np.linalg.norm(embedding)
                embedding = embedding.reshape(1, -1).astype('float32')

                similarity, position = index.search(embedding, 1)
                sim_score = similarity[0][0]
                pos       = position[0][0]

                if sim_score < THRESHOLD:
                    set_result(f"Not Registered ({sim_score:.2f})",
                               (0, 0, 255), (x1, y1, x2, y2), duration=2.0)
                else:
                    student_name   = names[pos]
                    display_name   = student_key_to_display(student_name)
                    already_marked = student_name in attendance_log
                    mark_attendance(student_name)

                    if already_marked:
                        label = f"Already Present: {display_name}"
                        color = (0, 200, 100)
                    else:
                        label = f"Present: {display_name} ({sim_score:.2f})"
                        color = (0, 255, 0)

                    set_result(label, color, (x1, y1, x2, y2), duration=2.0)

    # ── HUD + display ─────────────────────────────────────────────────────────
    draw_hud(frame)
    cv2.imshow("Face Attendance", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        spoof_checker.reset()
        print("🔄 Reset — ready for next person")

cap.release()
cv2.destroyAllWindows()

# ── Final attendance summary ──────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"Session ended. Total present: {len(attendance_log)}")
for pname, ptime in attendance_log.items():
    display = student_key_to_display(pname)
    print(f"  ✅ {display} — {ptime}")
print(f"{'='*40}")
print("Recognition stopped.")