from pathlib import Path
import glob
import os
import cv2
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from preprocess import apply_clahe, enhance_saturation
from tracking import CountSORTTracker, split_detections_by_side, parse_video_name


VIDEO_EXTENSIONS = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI")


def create_tracker(frame_rate: float, min_frames: int):
    return CountSORTTracker(
        frame_rate=frame_rate,
        minimum_iou_threshold=0.1,
        minimum_consecutive_frames=min_frames,
        track_activation_threshold=0.4,
        lost_track_buffer=10,
    )


def collect_video_paths(folder_path: str):
    folder = Path(folder_path)
    video_paths = []
    for pattern in VIDEO_EXTENSIONS:
        video_paths.extend(glob.glob(str(folder / pattern)))
    return sorted(video_paths)


def process_videos_folder(
    folder_path: str,
    output_video_dir: str,
    results_csv_path: str,
    model_loss_path: str,
    model_yield_path: str,
):
    """
    Process all videos in a folder:
    - run preprocessing on frames
    - run both models
    - split detections by side
    - track detections
    - save annotated videos
    - save per-video numeric results
    """
    output_video_dir = Path(output_video_dir)
    results_csv_path = Path(results_csv_path)

    output_video_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)

    model_loss = YOLO(model_loss_path)
    model_yield = YOLO(model_yield_path)

    results = []
    video_paths = collect_video_paths(folder_path)

    if not video_paths:
        raise FileNotFoundError(f"No videos found under: {folder_path}")

    for video_path in video_paths:
        try:
            tree, side_flag, plot = parse_video_name(video_path)
        except Exception as e:
            print(f"Skipping {video_path}, parse error: {e}")
            continue

        side_to_count = "l" if side_flag == "l" else "r"

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps or fps <= 0:
            fps = 30.0

        tracker_loss_l = create_tracker(frame_rate=fps, min_frames=3)
        tracker_loss_r = create_tracker(frame_rate=fps, min_frames=3)
        tracker_yield_l = create_tracker(frame_rate=fps, min_frames=4)
        tracker_yield_r = create_tracker(frame_rate=fps, min_frames=4)

        label_annotator_loss = sv.LabelAnnotator(color=sv.Color.RED, text_position=sv.Position.BOTTOM_CENTER)
        box_annotator_loss = sv.BoxAnnotator(color=sv.Color.RED)

        label_annotator_yield = sv.LabelAnnotator(color=sv.Color.GREEN, text_position=sv.Position.BOTTOM_CENTER)
        box_annotator_yield = sv.BoxAnnotator(color=sv.Color.GREEN)

        def callback(frame_bgr, frame_index: int):
            h, w = frame_bgr.shape[:2]

            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            clahe_image = apply_clahe(image_rgb)
            final_image = enhance_saturation(clahe_image)
            processed_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

            result_loss = model_loss(processed_bgr, conf=0.65, iou=0.3)[0]
            result_yield = model_yield(processed_bgr, conf=0.3)[0]

            detections_loss = sv.Detections.from_ultralytics(result_loss)
            detections_yield = sv.Detections.from_ultralytics(result_yield)

            det_loss_l, det_loss_r = split_detections_by_side(detections_loss, frame_width=w, margin_px=0)
            det_yield_l, det_yield_r = split_detections_by_side(detections_yield, frame_width=w, margin_px=0)

            tracked_loss_l = tracker_loss_l.update(det_loss_l)
            tracked_loss_r = tracker_loss_r.update(det_loss_r)
            tracked_yield_l = tracker_yield_l.update(det_yield_l)
            tracked_yield_r = tracker_yield_r.update(det_yield_r)

            if side_to_count == "l":
                total_yield = len(tracker_yield_l.all_track_ids)
                total_loss = len(tracker_loss_l.all_track_ids)
                side_str = "Left"
            else:
                total_yield = len(tracker_yield_r.all_track_ids)
                total_loss = len(tracker_loss_r.all_track_ids)
                side_str = "Right"

            out = processed_bgr.copy()

            loss_labels_l = [str(tid) for tid in tracked_loss_l.tracker_id] if tracked_loss_l.tracker_id is not None else None
            loss_labels_r = [str(tid) for tid in tracked_loss_r.tracker_id] if tracked_loss_r.tracker_id is not None else None
            yield_labels_l = [str(tid) for tid in tracked_yield_l.tracker_id] if tracked_yield_l.tracker_id is not None else None
            yield_labels_r = [str(tid) for tid in tracked_yield_r.tracker_id] if tracked_yield_r.tracker_id is not None else None

            out = label_annotator_loss.annotate(out, tracked_loss_l, labels=loss_labels_l)
            out = box_annotator_loss.annotate(out, tracked_loss_l)
            out = label_annotator_yield.annotate(out, tracked_yield_l, labels=yield_labels_l)
            out = box_annotator_yield.annotate(out, tracked_yield_l)

            out = label_annotator_loss.annotate(out, tracked_loss_r, labels=loss_labels_r)
            out = box_annotator_loss.annotate(out, tracked_loss_r)
            out = label_annotator_yield.annotate(out, tracked_yield_r, labels=yield_labels_r)
            out = box_annotator_yield.annotate(out, tracked_yield_r)

            cv2.putText(out, f"Total yield ({side_str} only): {total_yield}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(out, f"Total fruit loss ({side_str} only): {total_loss}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

            return out

        output_path = output_video_dir / Path(video_path).name

        sv.process_video(
            source_path=video_path,
            target_path=str(output_path),
            callback=callback,
        )

        if side_to_count == "l":
            yield_count = len(tracker_yield_l.all_track_ids)
            loss_count = len(tracker_loss_l.all_track_ids)
            side_label = "left"
        else:
            yield_count = len(tracker_yield_r.all_track_ids)
            loss_count = len(tracker_loss_r.all_track_ids)
            side_label = "right"

        results.append({
            "tree_number": tree,
            "side_of_filming": side_label,
            "yield_count": int(yield_count),
            "loss_count": int(loss_count),
            "plot": plot,
            "video_name": Path(video_path).name,
        })

        print(f"Done: {video_path}, yield={yield_count}, loss={loss_count}, side={side_label}")

    df = pd.DataFrame(results).sort_values(by=["plot", "tree_number", "video_name"])
    df.to_csv(results_csv_path, index=False)
    print(f"Saved results table to: {results_csv_path}")