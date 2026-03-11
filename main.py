from ultralytics import YOLO
from preprocess import preprocess_yolo_dataset
from detection_eval import evaluate_single_model
from visualize_images import visualize_predictions
from video_pipeline import process_videos_folder
from merge_results import merge_model_results_with_ground_truth
from analyze_counting_results import run_analysis

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

    process_videos_folder(
        folder_path="sample_videos/sample_videos_mishmar",
        output_video_dir="outputs/video_visualizations/mishmar",
        results_csv_path="outputs/tracking_results/mishmar_results.csv",
        model_loss_path="model_weights/loss_yolo11_weights.pt",
        model_yield_path="model_weights/yield_yolo11_weights.pt",
    )
    merged_df = merge_model_results_with_ground_truth(
        gt_csv_path="ground_truth/true_counts_yield_loss_orchards.csv",
        model_csv_paths=[
            "outputs/tracking_results/mishmar_results.csv",
            "outputs/tracking_results/tsora_results.csv",
        ],
        output_csv_path="outputs/final_results/merged_results.csv",
    )

    run_analysis(
        merged_csv_path="outputs/final_results/merged_results.csv",
    )

if __name__ == "__main__":
    main()