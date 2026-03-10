import os
import re
import numpy as np
import supervision as sv
from trackers import SORTTracker


class CountSORTTracker(SORTTracker):
    """Saves history of tracks to enable counting."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_track_ids = set()

    def update(self, detections: sv.Detections):
        active_tracks = super().update(detections)
        if active_tracks.tracker_id is not None:
            for i in range(len(active_tracks)):
                tid = active_tracks.tracker_id[i]
                self.all_track_ids.add(int(tid))
        return active_tracks


def split_detections_by_side(detections: sv.Detections, frame_width: int, margin_px: int = 0):
    """Split detections into left and right by bbox center to count by measuring tree anf fit aqusation geomatry."""
    if len(detections) == 0:
        return sv.Detections.empty(), sv.Detections.empty()

    half = frame_width / 2.0
    keep_l, keep_r = [], []

    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        cx = 0.5 * (x1 + x2)
        if cx <= (half - margin_px):
            keep_l.append(i)
        elif cx >= (half + margin_px):
            keep_r.append(i)

    det_l = detections[np.array(keep_l, dtype=int)] if keep_l else sv.Detections.empty()
    det_r = detections[np.array(keep_r, dtype=int)] if keep_r else sv.Detections.empty()
    return det_l, det_r


def parse_video_name(fname: str):
    """
    Example:
    1_r_mishmar_hanegev_11_org_vid_row_9_plot_11_1_first_forward.mp4
    Returns:
        tree (int), side (str), plot (str)
    """
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    parts = name.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected video name format: {base}")

    tree_str = parts[0]
    tree = int(re.sub(r"\D", "", tree_str))

    side = parts[1].lower()
    if side not in ("l", "r"):
        raise ValueError(f"Side should be 'l' or 'r' in: {base}")

    plot_parts = []
    i = 2
    while i < len(parts) and parts[i] != "org":
        plot_parts.append(parts[i])
        i += 1

    if not plot_parts:
        raise ValueError(f"Plot could not be identified from filename: {base}")

    plot = "_".join(plot_parts)
    return tree, side, plot