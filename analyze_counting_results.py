from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_mape(x_true, y_pred):
    """MAPE in percent, ignoring rows where GT is zero or NaN."""
    x_true = np.asarray(x_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = (x_true != 0) & (~np.isnan(x_true)) & (~np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_pred[mask] - x_true[mask]) / x_true[mask])) * 100.0


def add_plot_short_names(df: pd.DataFrame) -> pd.DataFrame:
    plot_mapping = {
        "mishmar_hanegev_11": "MH11",
        "mishmar_hanegev_7": "MH7",
    }
    df = df.copy()
    df["plot_short"] = df["plot"].map(plot_mapping).fillna(df["plot"])
    return df


def plot_regression(x, y, labels_df, x_label, y_label, title):
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    labels_df = labels_df[mask]

    if len(x) == 0:
        print(f"No data for: {title}")
        return

    m, b = np.polyfit(x, y, 1)
    y_pred_line = m * x + b

    ss_res = np.sum((y - y_pred_line) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    mae = np.mean(np.abs(y - x))
    bias = np.mean(y - x)
    mape = compute_mape(x, y)

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, alpha=0.7, label="Model output")

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, label="Regression line")
    plt.plot(x_line, x_line, linestyle="--", label="Perfect model (y = x)")

    for xx, yy, tn, ps in zip(x, y, labels_df["tree_number"], labels_df["plot_short"]):
        plt.text(xx, yy, f"{tn}, {ps}", fontsize=6, alpha=0.8)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    eq_text = (
        f"y = {m:.3f}x + {b:.3f}\n"
        f"R² = {r2:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"Bias = {bias:.3f}\n"
        f"MAPE = {mape:.2f}%"
    )

    plt.text(
        0.05, 0.95, eq_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.25),
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(x, y, x_label, title):
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x = x[mask].astype(float)
    y = y[mask].astype(float)

    if len(x) == 0:
        print(f"No data for: {title}")
        return

    errors = y - x
    mse = np.mean(errors ** 2)
    mean_err = np.mean(errors)
    std_err = np.std(errors)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, errors, alpha=0.7)
    plt.axhline(0, linestyle="--")

    plt.xlabel(x_label)
    plt.ylabel("Error (model - true)")
    plt.title(title)

    txt = (
        f"MSE = {mse:.3f}\n"
        f"Mean error = {mean_err:.3f}\n"
        f"Std error = {std_err:.3f}"
    )

    plt.text(
        0.05, 0.95, txt,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.25),
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_error_hist(x, y, title):
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x = x[mask].astype(float)
    y = y[mask].astype(float)

    if len(x) == 0:
        print(f"No data for: {title}")
        return

    errors = y - x

    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=20, alpha=0.7)
    plt.xlabel("Error (model - true)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_regression_per_plot(df_all, true_col, pred_col, title_prefix):
    for plot_name, sub in df_all.groupby("plot"):
        x = sub[true_col]
        y = sub[pred_col]
        ps = sub["plot_short"].iloc[0]

        mask = (~pd.isna(x)) & (~pd.isna(y))
        x = x[mask].astype(float)
        y = y[mask].astype(float)

        if len(x) < 2:
            continue

        m, b = np.polyfit(x, y, 1)
        y_pred_line = m * x + b
        ss_res = np.sum((y - y_pred_line) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, alpha=0.7, label="Model output")

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b
        plt.plot(x_line, y_line, label="Regression line")
        plt.plot(x_line, x_line, linestyle="--", label="Perfect model (y = x)")

        plt.xlabel(true_col)
        plt.ylabel(pred_col)
        plt.title(f"{title_prefix} - plot {ps}")

        eq_text = f"y = {m:.3f}x + {b:.3f}\nR² = {r2:.3f}"
        plt.text(
            0.05, 0.95, eq_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.25),
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def run_analysis(
    merged_csv_path: str,
    plots_output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Load merged results and create analysis plots.
    """
    df = pd.read_csv(merged_csv_path)
    df = add_plot_short_names(df)

    if plots_output_dir is not None:
        Path(plots_output_dir).mkdir(parents=True, exist_ok=True)

    plot_regression(
        x=df["loss_sum_true"],
        y=df["loss_sum_model"],
        labels_df=df,
        x_label="True loss counts (sum of 2 sides)",
        y_label="Model loss_count (sum of 2 sides)",
        title="Model loss_count vs True loss counts (sum of 2 sides)",
    )



    plot_regression(
        x=df["true yield counts"],
        y=df["max_yield"],
        labels_df=df,
        x_label="True yield counts",
        y_label="Model max_yield count",
        title="Model max_yield vs True yield counts",
    )



    plot_error_hist(
        x=df["loss_sum_true"],
        y=df["loss_sum_model"],
        title="Error distribution: loss_count - true loss counts",
    )

    plot_error_hist(
        x=df["true yield counts"],
        y=df["max_yield"],
        title="Error distribution: yield_avg - true yield counts",
    )

    return df