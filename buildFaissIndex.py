import faiss
import pickle
import numpy as np
import os

# ─── Step 1: Load embeddings.pkl ──────────────────────────────────────────────
print("Loading embeddings...")

with open("embeddings.pkl", 'rb') as f:
    embeddings_db = pickle.load(f)

print(f"Found {len(embeddings_db)} students: {list(embeddings_db.keys())}")

# ─── Step 2: Prepare vectors and names ────────────────────────────────────────
names = []      # ["Chaithra_G_L_10", "Chandana_H_N_23", ...]
vectors = []    # [[512 numbers], [512 numbers], ...]

for name, vector in embeddings_db.items():
    names.append(name)
    vectors.append(vector)

# Convert to numpy array
# Shape: (10, 512) → 10 students, 512 values each
vectors_array = np.array(vectors).astype('float32')
print(f"\nVectors shape: {vectors_array.shape}")
# Should print: (10, 512)

# ─── Step 3: Build FAISS index ────────────────────────────────────────────────
# IndexFlatIP = Flat Index using Inner Product (cosine similarity)
# 512 = size of each vector
dimension = 512
index = faiss.IndexFlatIP(dimension)

# Add all student vectors to FAISS
index.add(vectors_array)
print(f"FAISS index built with {index.ntotal} vectors")

# ─── Step 4: Save FAISS index ─────────────────────────────────────────────────
faiss.write_index(index, "faiss_index.bin")
print("✅ faiss_index.bin saved!")

# ─── Step 5: Save names list ──────────────────────────────────────────────────
with open("index_names.pkl", 'wb') as f:
    pickle.dump(names, f)
print("✅ index_names.pkl saved!")

# ─── Step 6: Verify everything ────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"✅ FAISS Index Built Successfully!")
print(f"   Total students in index: {index.ntotal}")
print(f"   Names saved: {names}")
print(f"{'='*50}")