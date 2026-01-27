from ultralytics import YOLO

def detect_players_first_frame(frame, model_path, conf=0.3):
    model = YOLO(model_path)
    result = model(frame, conf=conf)[0]

    detections = []
    next_id = 1

    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        if int(cls) == 0:  # assuming 0 = player
            detections.append((next_id, box.tolist()))
            next_id += 1

    return detections
