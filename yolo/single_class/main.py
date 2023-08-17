if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    model.train(data="config.yaml", epochs=400, patience=0, device=0, name='c-elegan_tracker', single_cls=True, batch=-1)
