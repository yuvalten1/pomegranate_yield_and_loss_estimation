from pathlib import Path
import shutil
import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to enhance local contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def enhance_saturation(image: np.ndarray, saturation_scale: float = 1.3) -> np.ndarray:
    """Enhance image saturation."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    s = np.clip(s.astype(np.float32) * saturation_scale, 0, 255).astype(np.uint8)

    merged = cv2.merge((h, s, v))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)


def preprocess_single_image(input_path: Path, output_path: Path, saturation_scale: float = 1.3) -> bool:
    """Read one image, preprocess it, and save it to output_path."""
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Failed to load: {input_path}")
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = apply_clahe(image_rgb)
    image_rgb = enhance_saturation(image_rgb, saturation_scale=saturation_scale)
    result_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_bgr)
    return True


def copy_labels(labels_src: Path, labels_dst: Path) -> int:
    """Copy YOLO label files from source to destination."""
    if not labels_src.exists():
        return 0

    labels_dst.mkdir(parents=True, exist_ok=True)
    copied = 0

    for label_path in labels_src.glob("*.txt"):
        shutil.copy2(label_path, labels_dst / label_path.name)
        copied += 1

    return copied


def process_split(split_src: Path, split_dst: Path, saturation_scale: float = 1.3) -> tuple[int, int]:
    """
    Process one YOLO split directory.
    Expected structure:
    split_src/
        images/
        labels/
    """
    images_src = split_src / "images"
    labels_src = split_src / "labels"

    images_dst = split_dst / "images"
    labels_dst = split_dst / "labels"

    if not images_src.exists():
        return 0, 0

    processed_images = 0

    for image_path in images_src.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            ok = preprocess_single_image(
                input_path=image_path,
                output_path=images_dst / image_path.name,
                saturation_scale=saturation_scale,
            )
            if ok:
                processed_images += 1

    copied_labels = copy_labels(labels_src, labels_dst)
    return processed_images, copied_labels


def preprocess_yolo_dataset(raw_dataset_dir: str, processed_dataset_dir: str, saturation_scale: float = 1.3) -> None:
    """
    Preprocess a YOLO dataset from raw_dataset_dir into processed_dataset_dir.

    Expected YOLO structure:
    dataset/
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
        data.yaml
    """
    raw_dataset_dir = Path(raw_dataset_dir)
    processed_dataset_dir = Path(processed_dataset_dir)

    if not raw_dataset_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dataset_dir}")

    total_images = 0
    total_labels = 0

    for split in ["train", "valid", "test"]:
        split_src = raw_dataset_dir / split
        split_dst = processed_dataset_dir / split

        images_count, labels_count = process_split(
            split_src=split_src,
            split_dst=split_dst,
            saturation_scale=saturation_scale,
        )

        total_images += images_count
        total_labels += labels_count

        if images_count > 0 or labels_count > 0:
            print(f"{split}: processed {images_count} images, copied {labels_count} labels")

    data_yaml_src = raw_dataset_dir / "data.yaml"
    data_yaml_dst = processed_dataset_dir / "data.yaml"
    if data_yaml_src.exists():
        data_yaml_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_yaml_src, data_yaml_dst)

    print(f"Done. Total processed images: {total_images}, total copied labels: {total_labels}")