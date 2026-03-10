from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv


def load_yolo_dataset(dataset_root: str) -> sv.DetectionDataset:
    """
    Load a YOLO dataset using supervision.
    Expected structure:
    dataset_root/
        train/
        valid/
        test/
        data.yaml
    """
    dataset_root = Path(dataset_root)

    return sv.DetectionDataset.from_yolo(
        images_directory_path=str(dataset_root / "test" / "images"),
        annotations_directory_path=str(dataset_root / "test" / "labels"),
        data_yaml_path=str(dataset_root / "data.yaml"),
    )


def make_inference_callback(model: YOLO):
    """Create a callback function for supervision benchmarks."""
    def callback(image: np.ndarray) -> sv.Detections:
        result = model(image)[0]
        return sv.Detections.from_ultralytics(result)

    return callback


def benchmark_model(
    model: YOLO,
    dataset: sv.DetectionDataset,
    title: str,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    normalize_plot: bool = False,
):
    """
    Run confusion matrix benchmark and mAP benchmark for a model on a dataset.
    """
    callback = make_inference_callback(model)

    confusion_matrix = sv.ConfusionMatrix.benchmark(
        dataset=dataset,
        callback=callback,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
    )

    confusion_matrix.plot(
        normalize=normalize_plot,
        title=title,
    )

    map_result = sv.MeanAveragePrecision.benchmark(
        dataset=dataset,
        callback=callback,
    )

    print(f"{title}")
    print(f"mAP50: {map_result.map50:.4f}")
    print(f"mAP50_95: {map_result.map50_95:.4f}")
    print(f"mAP75: {map_result.map75:.4f}")

    return confusion_matrix, map_result


def plot_normalized_confusion_matrix(confusion_matrix, title: str = "Normalized Confusion Matrix"):
    """
    Plot a normalized confusion matrix using matplotlib.
    """
    cm_data = confusion_matrix.matrix.astype(float)
    classes = confusion_matrix.classes

    row_sums = cm_data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized_matrix = cm_data / row_sums

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(normalized_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = normalized_matrix.max() / 2.0 if normalized_matrix.size > 0 else 0.5
    for i in range(normalized_matrix.shape[0]):
        for j in range(normalized_matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{normalized_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if normalized_matrix[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.show()


def calculate_metrics_from_confusion_matrix(confusion_matrix):
    """
    Calculate precision, recall, F1, and accuracy from the confusion matrix.
    Works for binary and multiclass cases.
    """
    matrix = confusion_matrix.matrix.astype(float)
    class_names = confusion_matrix.classes

    total = np.sum(matrix)
    if total == 0:
        print("Confusion matrix is empty.")
        return

    overall_accuracy = np.trace(matrix) / total
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    num_classes = matrix.shape[0]

    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):
        tp = matrix[i, i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        class_name = class_names[i] if i < len(class_names) else f"class_{i}"
        print(
            f'Class "{class_name}": '
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1 Score: {f1:.4f}"
        )

    print(f"Macro Precision: {np.mean(precisions):.4f}")
    print(f"Macro Recall: {np.mean(recalls):.4f}")
    print(f"Macro F1 Score: {np.mean(f1_scores):.4f}")


def evaluate_single_model(
    model_path: str,
    dataset_root: str,
    title: str,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
):
    """
    Full evaluation pipeline for one detection model on one YOLO dataset.
    """
    model = YOLO(model_path)
    dataset = load_yolo_dataset(dataset_root)

    confusion_matrix, map_result = benchmark_model(
        model=model,
        dataset=dataset,
        title=title,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        normalize_plot=False,
    )

    plot_normalized_confusion_matrix(
        confusion_matrix,
        title=f"{title}, Normalized",
    )

    print("Metrics:")
    calculate_metrics_from_confusion_matrix(confusion_matrix)

    return confusion_matrix, map_result