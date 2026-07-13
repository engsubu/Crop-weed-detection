from ultralytics import YOLO
import cv2
import argparse

def detect_and_spray(image_path, model_path):
    # Load trained model
    model = YOLO(model_path)
    
    # Run prediction
    results = model.predict(source=image_path, conf=0.4, save=True)
    
    # Smart Sprayer Logic
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)  # 0=crop, 1=weed
            conf = float(box.conf)
            if cls == 1:  # if weed detected
                print(f"WEED DETECTED with {conf:.2f} confidence -> SPRAY ON")
                # Here you would: GPIO.output(SPRAY_PIN, HIGH)
            else:
                print(f"CROP DETECTED with {conf:.2f} confidence -> SPRAY OFF")
                # GPIO.output(SPRAY_PIN, LOW)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='data/test.jpg')
    parser.add_argument('--model', type=str, default='models/crop_weed_yolov8n/weights/best.pt')
    args = parser.parse_args()
    
    detect_and_spray(args.img, args.model)
