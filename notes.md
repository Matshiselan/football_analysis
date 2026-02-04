## Multi-Object Tracking with Ultralytics YOLO

Got this from: https://docs.ultralytics.com/modes/track/#plotting-tracks-over-time

Ultralytics YOLO supports the following tracking algorithms. They can be enabled by passing the relevant YAML configuration file such as tracker=tracker_type.yaml:
•	BoT-SORT - Use botsort.yaml to enable this tracker.
•	ByteTrack - Use bytetrack.yaml to enable this tracker.

Bort sort yaml

```
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# BoT-SORT tracker defaults for mode="track"
# Docs: https://docs.ultralytics.com/modes/track/
tracker_type: botsort # (str) Tracker backend: botsort|bytetrack; choose botsort to enable BoT-SORT features
track_high_thresh: 0.25 # (float) First-stage match threshold; raise for cleaner tracks, lower to keep more
track_low_thresh: 0.1 # (float) Second-stage threshold for low-score matches; balances recovery vs drift
new_track_thresh: 0.25 # (float) Start a new track if no match ≥ this; higher reduces false tracks
track_buffer: 30 # (int) Frames to keep lost tracks alive; higher handles occlusion, increases ID switches risk
match_thresh: 0.8 # (float) Association similarity threshold (IoU/cost); tune with detector quality
fuse_score: True # (bool) Fuse detection score with motion/IoU for matching; stabilizes weak detections
# BoT-SORT specifics
gmc_method: sparseOptFlow # (str) Global motion compensation: sparseOptFlow|orb|none; helps moving camera scenes
# ReID model related thresh
proximity_thresh: 0.5 # (float) Min IoU to consider tracks proximate for ReID; higher is stricter
appearance_thresh: 0.8 # (float) Min appearance similarity for ReID; raise to avoid identity swaps
with_reid: False # (bool) Enable ReID model use; needs extra model and compute
model: auto # (str) ReID model name/path; "auto" uses detector features if available
```

## ByteTrack Yaml
```
tracker_type: bytetrack # (str) Tracker backend: botsort|bytetrack; choose bytetrack for the classic baseline
track_high_thresh: 0.25 # (float) First-stage match threshold; raise for cleaner tracks, lower to keep more
track_low_thresh: 0.1 # (float) Second-stage threshold for low-score matches; balances recovery vs drift
new_track_thresh: 0.25 # (float) Start a new track if no match ≥ this; higher reduces false tracks
track_buffer: 30 # (int) Frames to keep lost tracks alive; higher handles occlusion, increases ID switches risk
match_thresh: 0.8 # (float) Association similarity threshold (IoU/cost); tune with detector quality
fuse_score: True # (bool) Fuse detection score with motion/IoU for matching; stabilizes weak detections
```

## Enabling Re-Identification (ReID)
By default, ReID is turned off to minimize performance overhead. Enabling it is simple—just set with_reid: True in the tracker configuration. You can also customize the model used for ReID, allowing you to trade off accuracy and speed depending on your use case:
•	Native features (model: auto): This leverages features directly from the YOLO detector for ReID, adding minimal overhead. It's ideal when you need some level of ReID without significantly impacting performance. If the detector doesn't support native features, it automatically falls back to using yolo26n-cls.pt.
•	YOLO classification models: You can explicitly set a classification model (e.g. yolo26n-cls.pt) for ReID feature extraction. This provides more discriminative embeddings, but introduces additional latency due to the extra inference step.
For better performance, especially when using a separate classification model for ReID, you can export it to a faster backend like TensorRT:

```
from torch import nn
from ultralytics import YOLO
# Load the classification model
model = YOLO("yolo26n-cls.pt")
# Add average pooling layer
head = model.model.model[-1]
pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1))
pool.f, pool.i = head.f, head.i
model.model.model[-1] = pool
# Export to TensorRT
model.export(format="engine", half=True, dynamic=True, batch=32)
```

Once exported, you can point to the TensorRT model path in your tracker config, and it will be used for ReID during tracking.
