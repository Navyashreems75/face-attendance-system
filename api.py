"""
api.py  —  Sub-Group A  |  Face Attendance AI API
Runs on port 5001.
Sub-Group B's app.py on port 5000 calls this.

Endpoints:
  POST /recognize  — single frame recognition (base64 image + session_id)
  POST /enroll     — enroll new student (name + roll + base64 images)
  GET  /health     — quick check that API is running
  GET  /students   — list all enrolled students
  GET  /attendance/today — today's attendance log
"""

import os
import cv2
import faiss
import pickle
import base64
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from insightface.app import FaceAnalysis
from antispoofing import AntiSpoofing

# ─── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow Sub-Group B's app on port 5000 to call us

# ─── Paths ────────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE  = "embeddings.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
NAMES_FILE       = "index_names.pkl"
FACES_DIR        = "faces"

# ─── Recognition settings ─────────────────────────────────────────────────────
THRESHOLD = 0.65   # cosine similarity threshold — same as recognize.py

# ─── Load models ONCE at startup ─────────────────────────────────────────────
print("=" * 50)
print("Loading ArcFace model...")
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# SPEED FIX: 320×320 detector — faster on CPU, still reliable at kiosk distance
face_app.prepare(ctx_id=0, det_size=(320, 320))
print("✅ ArcFace loaded!")

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_FILE)
with open(NAMES_FILE, 'rb') as f:
    names = pickle.load(f)
print(f"✅ FAISS loaded — {index.ntotal} students enrolled!")

print("Loading embeddings DB...")
with open(EMBEDDINGS_FILE, 'rb') as f:
    embeddings_db = pickle.load(f)
print(f"✅ Embeddings loaded — {len(embeddings_db)} students!")

print("Loading Anti-Spoofing model...")
# SPEED FIX: load weights ONCE at startup into a shared instance.
# Per-session trackers are lightweight state dicts — no disk I/O per session.
_shared_spoof = AntiSpoofing(model_path='antispoofing_best.pth')
print("✅ Anti-Spoofing loaded!")

print("=" * 50)
print(f"API ready! Listening on port 5001")
print("=" * 50)


# ─── Attendance tracking ──────────────────────────────────────────────────────
attendance_log = {}   # { student_key: time_str }

# FIX: Per-session spoofing instances keyed by session_id from kiosk.
# Previously the code created a new AntiSpoofing instance per request
# (on every frame), which reset the blink/vote state each call.
# Now we maintain persistent session state across frames.
spoof_sessions = {}   # { session_id: AntiSpoofing instance }

def load_existing_attendance():
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
        print(f"ℹ️  {len(attendance_log)} student(s) already marked today.")

load_existing_attendance()


def mark_attendance(name):
    """Write attendance to CSV. Returns (newly_marked: bool, time_str: str)."""
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    filename = f"attendance_{date_str}.csv"

    if name in attendance_log:
        return False, attendance_log[name]

    file_exists = os.path.exists(filename)
    with open(filename, 'a') as f:
        if not file_exists:
            f.write("Name,Date,Time\n")
        f.write(f"{name},{date_str},{time_str}\n")

    attendance_log[name] = time_str
    print(f"✅ Attendance marked: {name} at {time_str}")
    return True, time_str


# ─── Helper: base64 image → cv2 frame ────────────────────────────────────────
def base64_to_frame(b64_string):
    """Convert a base64 encoded image to a BGR cv2 frame.
    Handles both raw base64 and data URI format."""
    try:
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"❌ base64_to_frame error: {e}")
        return None


# ─── Helper: rebuild FAISS index ─────────────────────────────────────────────
def rebuild_faiss_index():
    """Rebuild FAISS index from embeddings_db. Same as enrollNewStudent.py."""
    global index, names

    if not embeddings_db:
        # No students — write empty index
        index = faiss.IndexFlatIP(512)
        names = []
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open(NAMES_FILE, 'wb') as f:
            pickle.dump(names, f)
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(embeddings_db, f)
        return

    all_names      = list(embeddings_db.keys())
    all_embeddings = np.array(
        [embeddings_db[n] for n in all_names], dtype='float32'
    )

    new_index = faiss.IndexFlatIP(512)
    new_index.add(all_embeddings)
    index = new_index
    names = all_names

    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(NAMES_FILE, 'wb') as f:
        pickle.dump(names, f)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings_db, f)

    print(f"✅ FAISS rebuilt — {index.ntotal} students total.")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — /health
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status"           : "running",
        "students_enrolled": len(embeddings_db),
        "marked_today"     : len(attendance_log),
        "port"             : 5001
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — /recognize
# Called by Sub-Group B's /api/recognize route every SCAN_INTERVAL ms.
#
# Request JSON:
#   { "image": "data:image/jpeg;base64,...", "session_id": "kiosk-abc123" }
#
# FIX: session_id is now REQUIRED to maintain per-kiosk spoof state across
# frames. Without it, blink detection resets every frame and never passes.
#
# Response JSON examples:
#   { "status": "no_face" }
#   { "status": "liveness", "message": "Please blink", "progress": 0 }
#   { "status": "spoof", "message": "Spoof attempt detected" }
#   { "status": "present", "name": "Navya Shree", "confidence": 0.91,
#     "time": "09:32:11", "already_marked": false }
#   { "status": "unknown", "confidence": 0.43 }
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # FIX: use a meaningful session_id from the caller; fallback to 'default'
    session_id = (data.get('session_id') or 'default').strip()

    # Step 1: Decode base64 image
    frame = base64_to_frame(data['image'])
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Step 2: Detect faces with InsightFace
    # SPEED FIX: cap frame to 480px wide before detection. Larger frames give
    # no accuracy benefit with a 320x320 detector but cost significant CPU time.
    h_fr, w_fr = frame.shape[:2]
    if w_fr > 480:
        scale  = 480.0 / w_fr
        frame  = cv2.resize(frame, (480, int(h_fr * scale)), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces   = face_app.get(img_rgb)

    if not faces:
        return jsonify({"status": "no_face"})

    face = faces[0]

    # Step 3: Anti-spoofing — blink detection + MobileNetV2 voting
    # FIX: Create session only if it doesn't already exist.
    # Old code called AntiSpoofing() on every request which reset state.
    if session_id not in spoof_sessions:
        # FIX: lightweight per-session state — shared model weights, isolated state.
        # Old shallow-copy approach shared _vote_buffer and face_mesh between all
        # sessions, causing stale votes from previous people to corrupt new sessions.
        spoof_sessions[session_id] = {
            'eye_was_closed':  False,
            'blink_count':     0,
            'vote_buffer':     [],
            'vote_done':       False,
            'vote_result':     None,
            'settle_counter':  0,
            # Each session gets its own MediaPipe FaceMesh — sharing one
            # across sessions corrupts the landmark tracking state.
            'face_mesh': _shared_spoof.face_mesh.__class__(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ),
        }

    spoofer = spoof_sessions[session_id]

    # ── Blink detection using per-session MediaPipe FaceMesh ────────────────
    h_f, w_f = frame.shape[:2]
    rgb_full  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res       = spoofer['face_mesh'].process(rgb_full)

    if not res.multi_face_landmarks:
        return jsonify({"status": "liveness", "message": "Look directly at camera", "progress": 0})

    lms = res.multi_face_landmarks[0].landmark

    def _ear(indices):
        pts = [(lms[i].x * w_f, lms[i].y * h_f) for i in indices]
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        hh = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (v1 + v2) / (2.0 * hh) if hh > 0 else 0.3

    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]
    avg_ear   = (_ear(LEFT_EYE) + _ear(RIGHT_EYE)) / 2.0
    EAR_THRESH = 0.20

    if avg_ear < EAR_THRESH:
        spoofer['eye_was_closed'] = True
    elif spoofer['eye_was_closed']:
        spoofer['blink_count']   += 1
        spoofer['eye_was_closed'] = False

    if spoofer['blink_count'] == 0:
        return jsonify({"status": "liveness", "message": "Please blink once to verify", "progress": 0})

    # ── Anti-spoof voting using shared MobileNetV2 weights ───────────────────
    if spoofer['vote_done']:
        # Already decided — proceed to recognition or reject
        if not spoofer['vote_result']:
            spoof_sessions.pop(session_id, None)
            return jsonify({"status": "spoof", "message": "Spoof attempt detected"})
        # Fall through to recognition below
    else:
        # Settling phase — skip early post-blink frames
        if spoofer['settle_counter'] < _shared_spoof.SETTLE_FRAMES:
            spoofer['settle_counter'] += 1
            return jsonify({"status": "liveness", "message": "Hold still...", "progress": 0})

        # Padded face crop for MobileNetV2
        bbox            = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        x1p   = max(0, x1 - pad_x);  y1p = max(0, y1 - pad_y)
        x2p   = min(frame.shape[1], x2 + pad_x); y2p = min(frame.shape[0], y2 + pad_y)
        face_crop = frame[y1p:y2p, x1p:x2p]
        if face_crop.size == 0:
            return jsonify({"status": "no_face"})

        from PIL import Image as _PIL_Image
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor   = _shared_spoof.transform(
            _PIL_Image.fromarray(rgb_crop)
        ).unsqueeze(0).to(_shared_spoof.device)

        import torch as _torch
        with _torch.no_grad():
            probs     = _torch.softmax(_shared_spoof.model(tensor), dim=1)[0]
        real_prob = probs[1].item()
        voted_real = real_prob >= _shared_spoof.SPOOF_THRESHOLD
        spoofer['vote_buffer'].append(voted_real)
        print(f"[VOTE] session={session_id[-8:]} frame={len(spoofer['vote_buffer'])}/{_shared_spoof.VOTE_FRAMES}  real_prob={real_prob:.3f}  threshold={_shared_spoof.SPOOF_THRESHOLD}  voted={'REAL' if voted_real else 'FAKE'}")

        collected = len(spoofer['vote_buffer'])
        total     = _shared_spoof.VOTE_FRAMES

        if collected < total:
            progress = round(collected / total, 2)
            return jsonify({"status": "liveness", "message": "Verifying liveness...", "progress": progress})

        # Voting complete — make decision
        real_votes = sum(spoofer['vote_buffer'])
        ratio      = real_votes / collected
        spoofer['vote_done']   = True
        spoofer['vote_result'] = ratio >= _shared_spoof.REAL_VOTE_RATIO
        print(f"[VOTE] DECISION: {real_votes}/{collected} real votes ({ratio*100:.1f}%) — {'REAL ✅' if spoofer['vote_result'] else 'FAKE ❌'} (need {_shared_spoof.REAL_VOTE_RATIO*100:.0f}%)")

        if not spoofer['vote_result']:
            spoof_sessions.pop(session_id, None)
            return jsonify({"status": "spoof", "message": "Spoof attempt detected"})

    # ── Step 4: Liveness confirmed — run ArcFace recognition ─────────────────
    spoof_sessions.pop(session_id, None)   # clean up session for next person

    embedding = face.embedding
    embedding = embedding / np.linalg.norm(embedding)
    embedding = embedding.reshape(1, -1).astype('float32')

    # Step 5: Search FAISS index
    if index.ntotal == 0:
        return jsonify({"status": "unknown", "confidence": 0.0})

    similarity, position = index.search(embedding, 1)
    sim_score = float(similarity[0][0])
    pos       = int(position[0][0])

    if sim_score >= THRESHOLD:
        student_key  = names[pos]
        # FIX: display name strips trailing roll number (last underscore segment)
        parts        = student_key.rsplit('_', 1)
        display_name = parts[0].replace('_', ' ') if len(parts) == 2 else student_key
        newly_marked, time_str = mark_attendance(student_key)
        return jsonify({
            "status"        : "present",
            "name"          : display_name,
            "key"           : student_key,
            "confidence"    : round(sim_score, 4),
            "time"          : time_str,
            "already_marked": not newly_marked,
            "message"       : "Attendance already recorded for today" if not newly_marked
                              else "Attendance marked successfully"
        })
    else:
        return jsonify({
            "status"    : "unknown",
            "confidence": round(sim_score, 4),
            "message"   : "Face not recognised — not in database"
        })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — /enroll
# Called by app.py's /api/enroll route.
#
# Request JSON (from enroll.html via app.py):
#   { "name": "Arjun_Sharma_101", "roll": "101", "images": ["data:...", ...] }
#
# FIX: student_key is now taken DIRECTLY from the "name" field
# (which enroll.html already formats as "FirstName_LastName_Roll").
# Old code split on spaces and appended "_1", producing wrong keys like
# "Arjun_Sharma_101_1" instead of the expected "Arjun_Sharma_101".
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/enroll', methods=['POST'])
def enroll():
    data = request.get_json()

    if not data or 'name' not in data or 'images' not in data:
        return jsonify({"error": "name and images are required"}), 400

    # FIX: enroll.html sends name already formatted as "FirstName_LastName_Roll"
    # Use it directly as the student_key — don't re-process it.
    student_key = data['name'].strip().replace(' ', '_')
    roll        = data.get('roll', '').strip()
    images      = data['images']

    if not student_key:
        return jsonify({"error": "Name cannot be empty"}), 400
    if not images or len(images) < 5:
        return jsonify({"error": "At least 5 images required"}), 400

    # Save face images to disk (same folder structure as captureFaces.py)
    save_dir = os.path.join(FACES_DIR, student_key)
    os.makedirs(save_dir, exist_ok=True)

    embeddings = []
    photos_saved = 0
    warnings   = []

    for i, b64_img in enumerate(images[:10]):   # cap at 10
        frame = base64_to_frame(b64_img)
        if frame is None:
            warnings.append(f"Image {i+1}: could not decode")
            continue

        # Save photo to disk
        photo_path = os.path.join(save_dir, f"{i+1}.jpg")
        cv2.imwrite(photo_path, frame)
        photos_saved += 1

        # Extract embedding using InsightFace (RetinaFace + ArcFace)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces   = face_app.get(img_rgb)

        if not faces:
            warnings.append(f"Image {i+1}: no face detected, photo saved but skipped")
            continue

        # Use largest face in case of multiple
        best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb  = best.embedding
        emb  = (emb / np.linalg.norm(emb)).astype('float32')
        embeddings.append(emb)

    if not embeddings:
        return jsonify({
            "status"  : "failed",
            "error"   : "No face detected in any of the provided images. Retake with clear face visibility.",
            "warnings": warnings
        }), 400

    # Average all embeddings and normalise (same as generateEmbeddings.py)
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = (avg_emb / np.linalg.norm(avg_emb)).astype('float32')

    # Save to embeddings DB and rebuild FAISS
    embeddings_db[student_key] = avg_emb
    rebuild_faiss_index()

    parts        = student_key.rsplit('_', 1)
    display_name = parts[0].replace('_', ' ') if len(parts) == 2 else student_key

    print(f"✅ Enrolled: {display_name} ({student_key}) — {len(embeddings)} embeddings from {photos_saved} photos")

    resp = {
        "status"          : "enrolled",
        "embeddings_saved": True,
        "student_key"     : student_key,
        "name"            : display_name,
        "roll"            : roll,
        "photos_saved"    : photos_saved,
        "photos_used"     : len(embeddings),
        "total_students"  : len(embeddings_db),
        "redirect"        : "/dashboard"
    }
    if warnings:
        resp["warnings"] = warnings
    return jsonify(resp)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 — /students
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/students', methods=['GET'])
def students():
    student_list = []
    for key in sorted(embeddings_db.keys()):
        parts   = key.rsplit('_', 1)
        display = parts[0].replace('_', ' ') if len(parts) == 2 else key
        roll    = parts[1] if len(parts) == 2 else ''
        student_list.append({
            "key" : key,
            "name": display,
            "roll": roll
        })
    return jsonify({
        "students": student_list,
        "total"   : len(student_list)
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5 — /attendance/today
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/attendance/today', methods=['GET'])
def attendance_today():
    date_str     = datetime.now().strftime("%Y-%m-%d")
    present_list = []
    for key, time_str in attendance_log.items():
        parts   = key.rsplit('_', 1)
        display = parts[0].replace('_', ' ') if len(parts) == 2 else key
        present_list.append({
            "key" : key,
            "name": display,
            "time": time_str
        })
    return jsonify({
        "date"   : date_str,
        "present": present_list,
        "count"  : len(present_list)
    })


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=False, port=5001, host='0.0.0.0')