"""Download the AI face models (OpenCV Zoo, Apache-2.0 / MIT licensed) into models/.

    python download_models.py
"""
import os
import urllib.request

MODELS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

os.makedirs("models", exist_ok=True)
for name, url in MODELS.items():
    path = os.path.join("models", name)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        print("already present:", path)
        continue
    print("downloading", name, "...")
    urllib.request.urlretrieve(url, path)
    print("  saved", path, f"({os.path.getsize(path) // 1024} KB)")
print("Done.")
