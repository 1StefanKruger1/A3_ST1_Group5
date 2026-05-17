# Stage 1 EDA service for the macroinvertebrate image dataset.

from pathlib import Path
import json

import pandas as pd

from src.utils.plotting import (
    save_class_distribution_plot,
    save_image_size_distribution_plot,
    save_sample_grid,
)

# Create class 'EDAService'
class EDAService:
    #Generate summary statistics and visual outputs for Stage 1 EDA.

    def __init__(self, dataframe: pd.DataFrame, output_dir: Path) -> None:
        #Store the indexed dataset and output location.
        self.dataframe = dataframe.copy()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_dataframe(self) -> None:
        #Check that the dataframe contains the required EDA columns.
        required_columns = {"file_path", "label", "width", "height", "channels"}
        missing_columns = required_columns - set(self.dataframe.columns)

        if missing_columns:
            raise ValueError(
                "The EDA dataframe is missing these columns: "
                f"{sorted(missing_columns)}"
            )

        if self.dataframe.empty:
            raise ValueError("The EDA dataframe is empty. Check the dataset path.")

    def build_summary(self) -> dict[str, object]:
        #Return key dataset summary statistics.
        self.validate_dataframe()

        class_counts = self.dataframe["label"].value_counts()

        return {
            "total_images": int(len(self.dataframe)),
            "total_classes": int(self.dataframe["label"].nunique()),
            "mean_width": round(float(self.dataframe["width"].mean()), 2),
            "mean_height": round(float(self.dataframe["height"].mean()), 2),
            "min_width": int(self.dataframe["width"].min()),
            "max_width": int(self.dataframe["width"].max()),
            "min_height": int(self.dataframe["height"].min()),
            "max_height": int(self.dataframe["height"].max()),
            "most_common_class": str(class_counts.idxmax()),
            "most_common_class_count": int(class_counts.max()),
            "least_common_class": str(class_counts.idxmin()),
            "least_common_class_count": int(class_counts.min()),
        }

    def build_class_count_table(self) -> pd.DataFrame:
        #Return a table showing image counts and percentages by class.
        self.validate_dataframe()

        class_counts = self.dataframe["label"].value_counts().reset_index()
        class_counts.columns = ["label", "image_count"]

        total_images = len(self.dataframe)
        class_counts["percentage"] = (
            class_counts["image_count"] / total_images * 100
        ).round(2)

        return class_counts

    def save_summary_files(self) -> dict[str, Path]:
        #Save summary statistics and class counts to output files.
        self.validate_dataframe()

        summary = self.build_summary()
        class_counts = self.build_class_count_table()

        summary_json_path = self.output_dir / "dataset_summary.json"
        summary_csv_path = self.output_dir / "dataset_summary.csv"
        class_counts_path = self.output_dir / "class_counts.csv"

        summary_json_path.write_text(
            json.dumps(summary, indent=4),
            encoding="utf-8",
        )

        pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
        class_counts.to_csv(class_counts_path, index=False)

        return {
            "summary_json": summary_json_path,
            "summary_csv": summary_csv_path,
            "class_counts_csv": class_counts_path,
        }

    def save_class_distribution(self) -> Path:
        #Save a class-count chart for the dataset.
        self.validate_dataframe()

        return save_class_distribution_plot(
            dataframe=self.dataframe,
            output_path=self.output_dir / "class_distribution.png",
        )

    def save_image_size_distribution(self) -> Path:
        #Save width and height distribution charts.
        self.validate_dataframe()

        return save_image_size_distribution_plot(
            dataframe=self.dataframe,
            output_path=self.output_dir / "image_size_distribution.png",
        )

    def save_sample_images(self, sample_count: int = 9) -> Path:
        #Save a grid of representative sample images.
        self.validate_dataframe()

        return save_sample_grid(
            dataframe=self.dataframe,
            output_path=self.output_dir / "sample_images.png",
            sample_count=sample_count,
        )

    def run_all(self) -> dict[str, object]:
        #Run all Stage 1 EDA tasks and return output paths.
        self.validate_dataframe()

        outputs: dict[str, object] = {}

        outputs.update(self.save_summary_files())
        outputs["class_distribution"] = self.save_class_distribution()
        outputs["image_size_distribution"] = self.save_image_size_distribution()
        outputs["sample_images"] = self.save_sample_images()

        return outputs