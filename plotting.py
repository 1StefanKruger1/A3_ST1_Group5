#Plotting utilities for Stage 1 EDA and Stage 2 evaluation outputs.

# This module keeps visualisation code separate from service classes.
# That makes the project easier to test, maintain, and explain.

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import numpy as np

def ensure_output_dir(output_dir: Path) -> None:
    #Create an output directory if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)


def save_class_distribution_plot(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:
    #Save a bar chart showing the number of images in each class.
    if dataframe.empty:
        raise ValueError("Cannot plot class distribution because dataframe is empty.")

    ensure_output_dir(output_path.parent)

    plt.figure(figsize=(12, 6))
    order = dataframe["label"].value_counts().index

    sns.countplot(data=dataframe, x="label", order=order)

    plt.title("Macroinvertebrate Images per Class")
    plt.xlabel("Class label")
    plt.ylabel("Number of images")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_image_size_distribution_plot(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:
    #Save histograms showing image width and height distributions.
    if dataframe.empty:
        raise ValueError("Cannot plot image sizes because dataframe is empty.")

    ensure_output_dir(output_path.parent)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(dataframe["width"], bins=20, ax=axes[0])
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width in pixels")
    axes[0].set_ylabel("Number of images")

    sns.histplot(dataframe["height"], bins=20, ax=axes[1])
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height in pixels")
    axes[1].set_ylabel("Number of images")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_sample_grid(
    dataframe: pd.DataFrame,
    output_path: Path,
    sample_count: int = 9,
) -> Path:
    #Save a grid of sample images for quick visual inspection.
    if dataframe.empty:
        raise ValueError("Cannot create sample grid because dataframe is empty.")

    ensure_output_dir(output_path.parent)

    actual_count = min(sample_count, len(dataframe))
    sample_df = dataframe.sample(actual_count, random_state=42)

    columns = 3
    rows = (actual_count + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(10, 3.5 * rows))

    if rows == 1:
        axes = [axes] if columns == 1 else axes
    else:
        axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        image = cv2.imread(str(row["file_path"]))

        if image is None:
            ax.set_title("Unreadable image")
            ax.axis("off")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title(str(row["label"]))
        ax.axis("off")

    for ax in axes[actual_count:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_confusion_matrix_plot(
    confusion_matrix,
    labels: list[str],
    output_path: Path,
) -> Path:
    #Save a heatmap of the model confusion matrix.
    ensure_output_dir(output_path.parent)

    matrix_size = confusion_matrix.shape[0]

    if len(labels) != matrix_size:
        labels = [f"Class {index + 1}" for index in range(matrix_size)]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_training_report(
    report: str,
    output_path: Path,
) -> Path:
    #Save the classification report as a text file.
    ensure_output_dir(output_path.parent)
    output_path.write_text(report, encoding="utf-8")
    return output_path