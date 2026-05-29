"""
FaceAttend — enrollNewStudent.py
=================================
Flask server that receives images from enroll.html,
extracts embeddings with InsightFace, and updates:

    embeddings.pkl      – { "Name_Roll": avg_embedding (512-d float32) }
    faiss_index.bin     – FAISS IndexFlatIP
    index_names.pkl     – ordered names list matching FAISS rows
    faces/<Name_Roll>/  – raw JPEG photos

NOTE: In the full project this script's logic is absorbed into api.py.
Run this standalone ONLY if you want to test enrollment without app.py.

Run:
    python enrollNewStudent.py
Then open:
    http://localhost:5000/enroll
"""

import os, re, base64, pickle, shutil
import numpy as np
import faiss
import cv2
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from insightface.app import FaceAnalysis

# ── File paths ─────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE  = "embeddings.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
NAMES_FILE       = "index_names.pkl"
FACES_FOLDER     = "faces"

TEMPLATES_DIR = Path(__file__).parent / "templates"
Path(FACES_FOLDER).mkdir(exist_ok=True)

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")

# ── Load InsightFace once at startup ──────────────────────────────────────────
print("[INIT] Loading InsightFace buffalo_l ...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("[INIT] Model ready.")


# ── HELPERS ───────────────────────────────────────────────────────────────────

def safe_key(raw: str) -> str:
    """
    Sanitise the student key (name_roll string from enroll.html).
    FIX: The old regex stripped digits, which would destroy roll numbers.
    Now only strips truly illegal path/shell characters.
    Example: "Arjun_Sharma_101" → "Arjun_Sharma_101" (unchanged, correct)
    """
    # Allow letters, digits, underscores, hyphens, spaces only
    name = re.sub(r"[^\w\s\-]", "", raw).strip()
    # Collapse multiple spaces/underscores into single underscore
    return re.sub(r"[\s_]+", "_", name)


def is_already_enrolled(student_key: str) -> bool:
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            db = pickle.load(f)
        return student_key in db
    return False


def load_embeddings() -> dict:
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_and_rebuild(student_key: str, avg_embedding: np.ndarray) -> int:
    """Upsert avg_embedding into embeddings.pkl then rebuild FAISS index."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            embeddings_db = pickle.load(f)
    else:
        embeddings_db = {}

    embeddings_db[student_key] = avg_embedding

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings_db, f)

    names   = list(embeddings_db.keys())
    vectors = np.array(list(embeddings_db.values()), dtype="float32")

    index = faiss.IndexFlatIP(512)
    index.add(vectors)
    faiss.write_index(index, FAISS_INDEX_FILE)

    with open(NAMES_FILE, "wb") as f:
        pickle.dump(names, f)

    print(f"[FAISS] Rebuilt — {len(names)} students.")
    return len(embeddings_db)


def b64_to_bgr(b64: str):
    """Decode base64 image string → OpenCV BGR ndarray."""
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        buf = base64.b64decode(b64)
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[WARN] b64_to_bgr failed: {e}")
        return None


def extract_embedding(bgr: np.ndarray):
    """
    Run InsightFace on a BGR frame.
    Returns L2-normalised 512-d embedding or None if no face found.
    FIX: pick the LARGEST face (not just faces[0]) to handle cases where
    a small background face is detected before the main subject.
    """
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb)
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    emb  = best.embedding.astype("float32")
    emb  = emb / np.linalg.norm(emb)
    return emb


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/enroll", methods=["GET"])
def enroll_page():
    return send_from_directory(TEMPLATES_DIR, "enroll.html")


@app.route("/dashboard")
def dashboard():
    dash = TEMPLATES_DIR / "dashboard.html"
    if dash.exists():
        return send_from_directory(TEMPLATES_DIR, "dashboard.html")
    db = load_embeddings()
    return jsonify({"enrolled": list(db.keys()), "total": len(db)})


@app.route("/enroll", methods=["POST"])
def enroll_student():
    """
    Receives JSON from enroll.html:
    {
      "name":    "Arjun_Sharma_101",   ← already formatted by enroll.html
      "section": "A",
      "dept":    "CSE",
      "images":  ["<base64-jpeg>", ...]
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    raw_name   = (data.get("name") or "").strip()
    images_b64 = data.get("images") or []

    if not raw_name:
        return jsonify({"error": "'name' is required"}), 422
    if not isinstance(images_b64, list) or len(images_b64) < 5:
        return jsonify({"error": "At least 5 images required"}), 422

    images_b64  = images_b64[:10]
    section     = (data.get("section") or "").strip()
    dept        = (data.get("dept")    or "").strip()
    student_key = safe_key(raw_name)   # sanitise but preserve roll number

    student_folder = os.path.join(FACES_FOLDER, student_key)
    os.makedirs(student_folder, exist_ok=True)

    embeddings_list = []
    photos_saved    = 0
    warnings        = []

    for idx, b64 in enumerate(images_b64, start=1):
        bgr = b64_to_bgr(b64)
        if bgr is None:
            warnings.append(f"Image {idx}: could not decode")
            continue

        photo_path = os.path.join(student_folder, f"{idx}.jpg")
        cv2.imwrite(photo_path, bgr)
        photos_saved += 1

        emb = extract_embedding(bgr)
        if emb is None:
            warnings.append(f"Image {idx}: no face detected, photo saved but skipped")
            continue

        embeddings_list.append(emb)

    if not embeddings_list:
        return jsonify({
            "error":    "No face detected in any image. Retake with clear face visibility.",
            "warnings": warnings,
        }), 422

    all_embeddings = np.array(embeddings_list, dtype="float32")
    avg_embedding  = np.mean(all_embeddings, axis=0)
    avg_embedding  = avg_embedding / np.linalg.norm(avg_embedding)

    total = save_and_rebuild(student_key, avg_embedding)

    parts        = student_key.rsplit("_", 1)
    display_name = parts[0].replace("_", " ") if len(parts) == 2 else student_key
    roll         = parts[1] if len(parts) == 2 else ""

    print(f"[ENROLL] ✓ {display_name} (roll {roll}) | "
          f"photos={photos_saved} | embeddings={len(embeddings_list)} | total={total}")

    resp = {
        "ok":              True,
        "student_key":     student_key,
        "name":            display_name,
        "roll":            roll,
        "photos_saved":    photos_saved,
        "embeddings_used": len(embeddings_list),
        "total_enrolled":  total,
        # FIX: was "/dashboard" which doesn't exist in standalone mode.
        # Redirect to the enrollment page so the operator can enroll the next student.
        "redirect":        "/enroll",
    }
    if warnings:
        resp["warnings"] = warnings

    return jsonify(resp), 200


@app.route("/api/students", methods=["GET"])
def api_students():
    db = load_embeddings()
    students = []
    for key in db:
        parts = key.rsplit("_", 1)
        students.append({
            "key":  key,
            "name": parts[0].replace("_", " ") if len(parts) == 2 else key,
            "roll": parts[1] if len(parts) == 2 else "",
        })
    return jsonify({"students": students, "total": len(students)})


@app.route("/api/students/<path:student_key>", methods=["DELETE"])
def api_delete_student(student_key: str):
    student_key = safe_key(student_key)
    db_data = load_embeddings()

    if student_key not in db_data:
        return jsonify({"error": f"'{student_key}' not found"}), 404

    del db_data[student_key]
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(db_data, f)

    if db_data:
        names   = list(db_data.keys())
        vectors = np.array(list(db_data.values()), dtype="float32")
        index   = faiss.IndexFlatIP(512)
        index.add(vectors)
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open(NAMES_FILE, "wb") as f:
            pickle.dump(names, f)
    else:
        faiss.write_index(faiss.IndexFlatIP(512), FAISS_INDEX_FILE)
        with open(NAMES_FILE, "wb") as f:
            pickle.dump([], f)

    photo_dir = os.path.join(FACES_FOLDER, student_key)
    if os.path.exists(photo_dir):
        shutil.rmtree(photo_dir)

    print(f"[DELETE] Removed {student_key}. Remaining: {len(db_data)}")
    return jsonify({"ok": True, "deleted": student_key, "remaining": len(db_data)})


if __name__ == "__main__":
    print("=" * 56)
    print("  FaceAttend — enrollNewStudent.py (standalone)")
    print(f"  Embeddings : {EMBEDDINGS_FILE}")
    print(f"  FAISS      : {FAISS_INDEX_FILE}")
    print(f"  Names      : {NAMES_FILE}")
    print(f"  Faces      : {FACES_FOLDER}/")
    print("  URL        : http://localhost:5000/enroll")
    print("=" * 56)
    app.run(debug=True, host="0.0.0.0", port=5000)
