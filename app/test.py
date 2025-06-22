from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/mahjong/train/weights/best.pt")  

    model.predict(
        # source="/home/neolux/workspace/SmartMahjong/images_grey/",
        source="/home/neolux/workspace/SmartMahjong/dataset/mahjong_grey/images",
        imgsz=672,  # Input image size
        conf=0.1,  # Confidence threshold for predictions
        save=True,  # Save predictions to disk
        project="runs/mahjong",
    )
