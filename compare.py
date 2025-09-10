# main.py
import os
import shutil
import cv2
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
import tensorflow as tf

# === suppress warning TensorFlow ===
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

# === FastAPI app ===
app = FastAPI(title="Absensi Wajah API", version="1.0.0")

# === Path & Config ===
DATASET_PATH = "dataset"
COMPARE_PATH = "compare"
ABSENSI_FILE = "absensi.csv"
MAX_DISTANCE = 0.5  # semakin kecil semakin mirip (ArcFace default)

# === simpan absensi ===
def save_attendance(name: str, similarity: float, file: str = ABSENSI_FILE):
    now = datetime.now()
    waktu = now.strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.exists(file)
    with open(file, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Name,Similarity,Datetime\n")
        f.write(f"{name},{similarity:.2f}%,{waktu}\n")


# === endpoint upload foto untuk compare ===
@app.post("/compare")
async def compare(file: UploadFile = File(...)):
    # pastikan folder compare ada
    if not os.path.exists(COMPARE_PATH):
        os.makedirs(COMPARE_PATH)

    # bersihkan file lama di compare biar tidak numpuk
    for old_file in os.listdir(COMPARE_PATH):
        os.remove(os.path.join(COMPARE_PATH, old_file))

    # simpan file baru
    file_path = os.path.join(COMPARE_PATH, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = DeepFace.find(
            img_path=file_path,
            db_path=DATASET_PATH,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="opencv"
        )

        if len(result[0]) > 0:
            df = (
                result[0]
                .copy()
                .sort_values(by="distance", ascending=True)
                .reset_index(drop=True)
            )

            best_identity = df.iloc[0]["identity"]
            best_distance = df.iloc[0]["distance"]
            best_name = os.path.splitext(os.path.basename(best_identity))[0]

            # konversi ke similarity %
            similarity = (1 - best_distance) * 100
            similarity = max(0, min(similarity, 100))

            if best_distance <= MAX_DISTANCE:
                save_attendance(best_name, similarity)
                return JSONResponse({
                    "status": "match",
                    "name": best_name,
                    "similarity": round(similarity, 2),
                    "compare_file": file.filename
                })
            else:
                return JSONResponse({
                    "status": "not_match",
                    "name": best_name,
                    "similarity": round(similarity, 2),
                    "compare_file": file.filename
                })
        else:
            return JSONResponse({"status": "no_candidate", "compare_file": file.filename})

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

