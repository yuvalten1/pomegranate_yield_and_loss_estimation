from pathlib import Path
import numpy as np
import pandas as pd


def merge_model_results_with_ground_truth(
    gt_csv_path: str,
    model_csv_paths: list[str],
    output_csv_path: str,
) -> pd.DataFrame:
    """
    Merge model result CSV files with ground-truth CSV and compute
    tree-level aggregated metrics for yield and loss.

    Parameters
    ----------
    gt_csv_path : str
        Path to ground-truth CSV.
    model_csv_paths : list[str]
        Paths to one or more model result CSV files.
    output_csv_path : str
        Path to save merged results CSV.

    Returns
    -------
    pd.DataFrame
        The merged dataframe.
    """
    model_dfs = [pd.read_csv(path) for path in model_csv_paths]
    df_model_all = pd.concat(model_dfs, ignore_index=True)

    df_gt = pd.read_csv(gt_csv_path)
    df_gt_clean = df_gt[["video_name", "true yield counts", "true loss counts"]]

    df_merged = pd.merge(
        df_model_all,
        df_gt_clean,
        on="video_name",
        how="inner",
    )

    df_merged["yield_sum"] = np.nan
    df_merged["yield_avg"] = np.nan
    df_merged["max_yield"] = np.nan

    df_merged["loss_sum_model"] = np.nan
    df_merged["loss_sum_true"] = np.nan

    df_merged["yield_ratio_pred_over_gt"] = np.nan
    df_merged["yield_error_pct"] = np.nan
    df_merged["loss_ratio_pred_over_gt"] = np.nan
    df_merged["loss_error_pct"] = np.nan

    grouped = df_merged.groupby(["tree_number", "plot"], sort=False)

    for (tree, plot), idx in grouped.groups.items():
        rows = list(idx)

        if len(rows) == 1:
            i = rows[0]

            y = float(df_merged.loc[i, "yield_count"])
            df_merged.loc[i, "yield_sum"] = y
            df_merged.loc[i, "yield_avg"] = y
            df_merged.loc[i, "max_yield"] = y

            lm = float(df_merged.loc[i, "loss_count"])
            lt = float(df_merged.loc[i, "true loss counts"])
            df_merged.loc[i, "loss_sum_model"] = lm
            df_merged.loc[i, "loss_sum_true"] = lt

            y_pred = float(df_merged.loc[i, "max_yield"])
            y_gt = float(df_merged.loc[i, "true yield counts"])
            if y_gt != 0:
                df_merged.loc[i, "yield_ratio_pred_over_gt"] = y_pred / y_gt
                df_merged.loc[i, "yield_error_pct"] = 100.0 * (y_pred - y_gt) / y_gt

            l_pred = float(df_merged.loc[i, "loss_sum_model"])
            l_gt = float(df_merged.loc[i, "true loss counts"])
            if l_gt != 0:
                df_merged.loc[i, "loss_ratio_pred_over_gt"] = l_pred / l_gt
                df_merged.loc[i, "loss_error_pct"] = 100.0 * (l_pred - l_gt) / l_gt

        elif len(rows) == 2:
            i1, i2 = rows

            y1 = float(df_merged.loc[i1, "yield_count"])
            y2 = float(df_merged.loc[i2, "yield_count"])
            total_y = y1 + y2
            avg_y = total_y / 2.0
            max_y = max(y1, y2)

            df_merged.loc[i1, "yield_sum"] = total_y
            df_merged.loc[i1, "yield_avg"] = avg_y
            df_merged.loc[i1, "max_yield"] = max_y

            lm1 = float(df_merged.loc[i1, "loss_count"])
            lm2 = float(df_merged.loc[i2, "loss_count"])
            lt1 = float(df_merged.loc[i1, "true loss counts"])
            lt2 = float(df_merged.loc[i2, "true loss counts"])

            total_lm = lm1 + lm2
            total_lt = lt1 + lt2

            df_merged.loc[i1, "loss_sum_model"] = total_lm
            df_merged.loc[i1, "loss_sum_true"] = total_lt

            y_pred = float(df_merged.loc[i1, "max_yield"])
            y_gt = float(df_merged.loc[i1, "true yield counts"])
            if y_gt != 0:
                df_merged.loc[i1, "yield_ratio_pred_over_gt"] = y_pred / y_gt
                df_merged.loc[i1, "yield_error_pct"] = 100.0 * (y_pred - y_gt) / y_gt

            l_pred = float(df_merged.loc[i1, "loss_sum_model"])
            l_gt = float(df_merged.loc[i1, "loss_sum_true"])
            if l_gt != 0:
                df_merged.loc[i1, "loss_ratio_pred_over_gt"] = l_pred / l_gt
                df_merged.loc[i1, "loss_error_pct"] = 100.0 * (l_pred - l_gt) / l_gt

        else:
            print(f"Warning: tree {tree}, plot {plot} has {len(rows)} rows, skipping extra logic")

    yield_mape = df_merged["yield_error_pct"].abs().mean()
    loss_mape = df_merged["loss_error_pct"].abs().mean()

    print(f"Yield MAPE (%): {yield_mape:.2f}")
    print(f"Loss MAPE (%): {loss_mape:.2f}")

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_merged.to_csv(output_csv_path, index=False)
    print(f"Saved results table to: {output_csv_path}")

    return df_merged