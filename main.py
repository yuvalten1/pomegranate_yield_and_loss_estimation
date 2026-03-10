from ultralytics import YOLO
from preprocess import preprocess_yolo_dataset
from detection_eval import evaluate_single_model
from visualize_images import visualize_predictions
from video_pipeline import process_videos_folder


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
    model_yield = YOLO("model_weights/yield_yolo11_weights.pt")
    model_loss = YOLO("model_weights/loss_yolo11_weights.pt")

    visualize_predictions(
        test_images_path="sample_datasets/processed/sample_dataset_yield_processed/test/images",
        model_yield=model_yield,
        model_loss=model_loss,
        output_dir="outputs/visualizations/detection_predictions_images",
        show=False,
        max_images=20,
    )

    process_videos_folder(
        folder_path="sample_videos/sample_videos_tsora",
        output_video_dir="outputs/video_visualizations/tsora",
        results_csv_path="outputs/tracking_results/tsora_results.csv",
        model_loss_path="model_weights/loss_yolo11_weights.pt",
        model_yield_path="model_weights/yield_yolo11_weights.pt",
    )


if __name__ == "__main__":
    main()