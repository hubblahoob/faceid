import cv2
import base64
import json
import requests
from datetime import datetime

API_URL = "http://localhost:8888/compare_base64"

def capture_and_send():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam tidak ditemukan")
        return

    print("Tekan 'SPACE' untuk capture, 'ESC' untuk keluar...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame dari kamera")
            break

        cv2.imshow("Capture & Send to API", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

        elif key == 32:  # SPACE
            # Encode ke base64
            _, buffer = cv2.imencode(".jpg", frame)
            image_b64 = base64.b64encode(buffer).decode("utf-8")

            payload = {"image": image_b64}

            # Simpan ke file JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"payload_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
            print(f"✅ Payload base64 tersimpan di {filename}")

            # Kirim ke API
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    print("=== HASIL DARI API ===")
                    print(response.json())
                else:
                    print(f"⚠️ Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"⚠️ Gagal request ke API: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_send()
