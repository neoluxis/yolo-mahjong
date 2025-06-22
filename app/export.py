from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/mahjong/train/weights/best.pt")  # Load the trained model

    model.export(
        format="onnx",  # Export format
        imgsz=672,  # Input image size
        simplify=True,  # Simplify the exported model
        optimize=True,  # Optimize the exported model
        project="runs/mahjong",  # Project directory for saving the exported model
    )