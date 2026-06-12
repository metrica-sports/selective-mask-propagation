"""Selective mask propagation parameters — single source of truth.

Paper-named parameters (Section 3 of the paper):

    MARGIN_ENTRY      τ_entry   open a window when a matched detection's
                                assignment margin drops below this
    MARGIN_EXIT       τ_exit    margin must recover above this at exit
    SEED_MARGIN       τ_seed    minimum margin for seed-frame candidates
    SEED_CONSECUTIVE  N_seed    consecutive clean frames required to seed
    EXIT_CONSECUTIVE  N_exit    consecutive frames exit conditions must hold
    IOMA_EXIT         τ_IoMA    mask-in-box containment for a track match

Implementation knobs (not named in the paper):

    SEED_CLEAN_IOU      bbox-overlap threshold for "spatially isolated"
                        (used at seeding and in the exit isolation checks)
    MASK_OVERLAP_EXIT   pixel-IoU at which two masks count as converged
                        (sustained convergence discards both as DEGRADED)
    AREA_DEGRADATION    mask discarded at exit if its area shrank below
                        this fraction of its seed area
    GAP_TRIGGER         frames a track must be absent before its
                        reappearance opens a gap window
    BORDER_MARGIN       px from frame edge at which the base tracker
                        kills a track instead of keeping it in the lost
                        buffer
    SAM_BORDER_MARGIN   px from frame edge at which a mask counts as
                        touching the border (tighter masks need a wider
                        margin than bboxes)
"""

MARGIN_ENTRY = 0.05
MARGIN_EXIT = 0.10
SEED_MARGIN = 0.10
SEED_CONSECUTIVE = 5
EXIT_CONSECUTIVE = 5
IOMA_EXIT = 0.80

SEED_CLEAN_IOU = 0.10
MASK_OVERLAP_EXIT = 0.90
AREA_DEGRADATION = 0.25
GAP_TRIGGER = 7
BORDER_MARGIN = 10
SAM_BORDER_MARGIN = 20
