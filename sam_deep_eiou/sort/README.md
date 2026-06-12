# SORT Tracker

Ported from [SORT](https://github.com/abewley/sort). Kalman filter + IoU for association. No modifications to the tracking logic.

Additions for SAM-Deep-EIoU integration:
- Assignment margin computation from the IoU cost matrix
- Edge-kill (tracks at frame border are removed instead of kept in lost buffer)
- Replaced filterpy dependency with inline Kalman filter
- `update()` returns tracker objects with `.track_id`, `.last_tlbr`, `.margin`
