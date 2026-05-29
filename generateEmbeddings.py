import insightface
import cv2
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis

# ─── Step 1: Load the ArcFace model ───────────────────────────────────────────
print("Loading ArcFace model...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded successfully!")

# ─── Step 2: Define folders ────────────────────────────────────────────────────
FACES_FOLDER = "faces"        # folder where your photos are saved
OUTPUT_FILE  = "embeddings.pkl"  # where we save the final embeddings

# ─── Step 3: Loop through each student folder ─────────────────────────────────
embeddings_db = {}  # dictionary: {student_name: average_embedding_vector}

student_folders = os.listdir(FACES_FOLDER)
print(f"\nFound {len(student_folders)} students: {student_folders}")

for student_folder in student_folders:
    student_path = os.path.join(FACES_FOLDER, student_folder)
    
    if not os.path.isdir(student_path):
        continue  # skip if not a folder
    
    print(f"\nProcessing: {student_folder}")
    
    all_embeddings = []  # collect all embeddings for this student
    
    # ─── Step 4: Loop through each photo ──────────────────────────────────────
    photo_files = os.listdir(student_path)
    
    for photo_file in photo_files:
        photo_path = os.path.join(student_path, photo_file)
        
        # Read the image
        img = cv2.imread(photo_path)
        if img is None:
            print(f"  Skipping {photo_file} - could not read")
            continue
        
        # Convert BGR to RGB (insightface needs RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ─── Step 5: Detect face and get embedding ─────────────────────────
        faces = app.get(img_rgb)
        
        if len(faces) == 0:
            print(f"  No face found in {photo_file} - skipping")
            continue
        
        if len(faces) > 1:
            print(f"  Multiple faces in {photo_file} - using largest")
        
        # Get the first (largest) face's embedding
        embedding = faces[0].embedding  # this is the 512-dim vector
        all_embeddings.append(embedding)
        print(f"  ✅ {photo_file} → embedding shape: {embedding.shape}")
    
    # ─── Step 6: Average all embeddings for this student ──────────────────────
    if len(all_embeddings) == 0:
        print(f"  ❌ No valid embeddings for {student_folder} - skipping")
        continue
    
    avg_embedding = np.mean(all_embeddings, axis=0)
    
    # Normalize the average embedding (important for cosine similarity later)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    embeddings_db[student_folder] = avg_embedding
    print(f"  ✅ Final embedding saved for {student_folder}")
    print(f"     Photos processed: {len(all_embeddings)}/{len(photo_files)}")

# ─── Step 7: Save all embeddings to a file ────────────────────────────────────
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(embeddings_db, f)

print(f"\n{'='*50}")
print(f"✅ Done! Embeddings saved to {OUTPUT_FILE}")
print(f"   Total students enrolled: {len(embeddings_db)}")
for name in embeddings_db:
    print(f"   → {name}: vector shape {embeddings_db[name].shape}")
print(f"{'='*50}")