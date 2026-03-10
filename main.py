from preprocess import preprocess_yolo_dataset


def main():
    preprocess_yolo_dataset(
        raw_dataset_dir="sample_datasets/raw/sample_dataset_yield_raw",
        processed_dataset_dir="sample_datasets/processed/sample_dataset_yield_processed",
    )

    preprocess_yolo_dataset(
        raw_dataset_dir="sample_datasets/raw/sample_dataset_loss_raw",
        processed_dataset_dir="sample_datasets/processed/sample_dataset_loss_processed",
    )


if __name__ == "__main__":
    main()