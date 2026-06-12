# Deep-EIoU Tracker

Ported from [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU). Kalman filter + expanded IoU + embedding cosine similarity for association.

Additions for SAM-Deep-EIoU integration:
- Assignment margin computation from the cost matrix in all three association rounds (`second_best - best` per matched column), exposed as `.margin` on each track
- Edge-kill (tracks at frame border are removed instead of kept in lost buffer)
- `step_track()` entry point: takes per-frame detections + embeddings, returns `(tracks, margins)` dicts
