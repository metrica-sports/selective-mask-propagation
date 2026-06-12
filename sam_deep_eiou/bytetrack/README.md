# ByteTrack

Ported from [ByteTrack](https://github.com/ifzhang/ByteTrack). Two-round IoU association (high + low confidence detections) with Kalman filter. No modifications to the tracking logic.

Reuses basetrack, kalman_filter, and matching from deep_eiou (same original codebase).

Additions for SAM-Deep-EIoU integration:
- Assignment margin computation from the IoU cost matrix
- Edge-kill (tracks at frame border are removed instead of kept in lost buffer)
- Simplified update() signature: takes (dets, scores) instead of (output_results, img_info, img_size)
- Returns tracker objects with .track_id, .last_tlbr, .margin
