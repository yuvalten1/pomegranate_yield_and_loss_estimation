from pathlib import Path
import cv2
from ultralytics import YOLO
import supervision as sv


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def visualize_predictions(
    test_images_path: str,
    model_yield: YOLO,
    model_loss: YOLO,
    output_dir: str | None = None,
    show: bool = False,
    max_images: int | None = None,
) -> None:
    """
    Run both detection models on images, annotate predictions, and optionally save results.

    Green boxes = yield model
    Red boxes = loss model
    """
    test_images_path = Path(test_images_path)

    if not test_images_path.exists():
        raise FileNotFoundError(f"Test images path not found: {test_images_path}")

    image_paths = [
        p for p in test_images_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if max_images is not None:
        image_paths = image_paths[:max_images]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    bbox_annotator_yield = sv.BoundingBoxAnnotator(color=sv.Color.GREEN, thickness=3)
    bbox_annotator_loss = sv.BoundingBoxAnnotator(color=sv.Color.RED, thickness=3)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path}")
            continue

        results_yield = model_yield(image)[0]
        results_loss = model_loss(image)[0]

        detections_yield = sv.Detections.from_ultralytics(results_yield)
        detections_loss = sv.Detections.from_ultralytics(results_loss)

        annotated_image = bbox_annotator_yield.annotate(
            scene=image.copy(),
            detections=detections_yield,
        )

        annotated_image = bbox_annotator_loss.annotate(
            scene=annotated_image,
            detections=detections_loss,
        )

        if output_dir is not None:
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), annotated_image)

        if show:
            sv.plot_image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    print(f"Processed {len(image_paths)} images")