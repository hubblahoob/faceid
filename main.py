from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
from deepface import DeepFace
from datetime import datetime
import os, base64

app = FastAPI(title="Absensi Wajah API", version="1.0.0")

DATASET_PATH = "dataset"
COMPARE_PATH = "compare"
ABSENSI_FILE = "absensi.csv"
THRESHOLD = 60  # minimal similarity dianggap cocok


# === Fungsi simpan absensi ke CSV ===
def save_attendance(name, similarity, file=ABSENSI_FILE):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file, "a", encoding="utf-8") as f:
        f.write(f"{name},{similarity:.2f}%,{now}\n")


# === Endpoint root ===
@app.get("/")
async def root():
    return {"message": "API Absensi Wajah aktif 🚀"}


# === Endpoint compare via FILE (form-data) ===
@app.post("/compare")
async def compare_face_file(file: UploadFile = File(...)):
    try:
        os.makedirs(COMPARE_PATH, exist_ok=True)
        file_location = os.path.join(COMPARE_PATH, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        result = DeepFace.find(
            img_path=file_location,
            db_path=DATASET_PATH,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="opencv"
        )

        if len(result[0]) > 0:
            df = result[0].sort_values(by="distance", ascending=True).reset_index(drop=True)
            best_identity = df.iloc[0]["identity"]
            best_distance = df.iloc[0]["distance"]

            best_name = os.path.splitext(os.path.basename(best_identity))[0]
            similarity = (1 - best_distance) * 100
            similarity = max(0, min(similarity, 100))

            status = "COCOK" if similarity >= THRESHOLD else "TIDAK COCOK"

            if status == "COCOK":
                save_attendance(best_name, similarity)

            return {
                "dataset": best_name,
                "similarity": f"{similarity:.2f}%",
                "status": status
            }
        else:
            return {"status": "Tidak ada kandidat di dataset"}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# === Endpoint compare via BASE64 (JSON) ===
@app.post("/compare_base64")
async def compare_face_base64(payload: dict = Body(...)):
    try:
        image_b64 = payload.get("image")
        if not image_b64:
            return JSONResponse({"error": "Base64 image tidak ditemukan"}, status_code=400)

        os.makedirs(COMPARE_PATH, exist_ok=True)
        file_location = os.path.join(COMPARE_PATH, "temp_image.jpg")

        image_bytes = base64.b64decode(image_b64)
        with open(file_location, "wb") as f:
            f.write(image_bytes)

        result = DeepFace.find(
            img_path=file_location,
            db_path=DATASET_PATH,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="opencv"
        )

        if len(result[0]) > 0:
            df = result[0].sort_values(by="distance", ascending=True).reset_index(drop=True)
            best_identity = df.iloc[0]["identity"]
            best_distance = df.iloc[0]["distance"]

            best_name = os.path.splitext(os.path.basename(best_identity))[0]
            similarity = (1 - best_distance) * 100
            similarity = max(0, min(similarity, 100))

            status = "COCOK" if similarity >= THRESHOLD else "TIDAK COCOK"

            if status == "COCOK":
                save_attendance(best_name, similarity)

            return {
                "dataset": best_name,
                "similarity": f"{similarity:.2f}%",
                "status": status
            }
        else:
            return {"status": "Tidak ada kandidat di dataset"}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# === Endpoint lihat absensi ===
@app.get("/attendance")
async def get_attendance():
    if not os.path.exists(ABSENSI_FILE):
        return {"message": "Belum ada absensi"}

    with open(ABSENSI_FILE, "r", encoding="utf-8") as f:
        data = f.read().splitlines()

    return {"attendance": data}
