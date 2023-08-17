if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO("yolov8n-seg.pt")

    model.train(data="config.yaml", epochs=450, patience=0, device=0, batch=1)
