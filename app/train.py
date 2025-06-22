from ultralytics import YOLO

num_epochs = 300

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8s.pt")  # Load a pretrained model
    # model = YOLO("runs/mahjong/train/weights/best.pt")
    
    model.train(
        data="app/dataset/mahjong-foss-1.yaml",  # Path to dataset configuration file
        epochs=num_epochs,  # Number of training epochs
        imgsz=672,  # Input image size
        batch=128,  # Batch size
        workers=24,  # Number of workers for data loading
        project="runs/mahjong",  # Project name for saving results
    )