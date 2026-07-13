from ultralytics import YOLO

def main():
    # Load YOLOv8n model. 'n' = nano, fastest to train
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=512,  # same as your dataset
        batch=16,
        project='models',
        name='crop_weed_yolov8n',
        patience=10,  # early stopping
        plots=True
    )

    print("Training complete! Best weights saved in models/crop_weed_yolov8n/weights/best.pt")

if __name__ == "__main__":
    main()
