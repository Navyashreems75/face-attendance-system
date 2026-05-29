import os, glob, shutil

FACES_DIR  = "faces"           # your existing faces folder
OUTPUT_DIR = "my_real_faces"   # output folder to zip and upload

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0
for root, dirs, files in os.walk(FACES_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(root, f)
            dst = os.path.join(OUTPUT_DIR, f"real_{count:04d}.jpg")
            shutil.copy(src, dst)
            count += 1

print(f"✅ Copied {count} real face images to {OUTPUT_DIR}/")