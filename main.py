from preprocess import preprocess_yolo_dataset
from detection_eval import evaluate_single_model


def main():
    preprocess_yolo_dataset(
        raw_dataset_dir="sample_datasets/raw/sample_dataset_yield_raw",
        processed_dataset_dir="sample_datasets/processed/sample_dataset_yield_processed",
    )

    preprocess_yolo_dataset(
        raw_dataset_dir="sample_datasets/raw/sample_dataset_loss_raw",
        processed_dataset_dir="sample_datasets/processed/sample_dataset_loss_processed",
    )
    evaluate_single_model(
        model_path="model_weights/yield_yolo11_weights.pt",
        dataset_root="sample_datasets/processed/sample_dataset_yield_processed",
        title="Yield Model Confusion Matrix",
    )

    evaluate_single_model(
        model_path="model_weights/loss_yolo11_weights.pt",
        dataset_root="sample_datasets/processed/sample_dataset_loss_processed",
        title="Loss Model Confusion Matrix",
    )

if __name__ == "__main__":
    main()