if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    model.train(data="config.yaml", epochs=200, patience=0, device=0, name='coloured_detection', batch=8)
