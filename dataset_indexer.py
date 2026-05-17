import cv2
import pandas as pd
from src.config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS


class DatasetIndexer:
    def __init__(self):
        self.records = []


    def build_dataframe(self):
        self.records = []

        for file_path in RAW_DATA_DIR.rglob('*'):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        img = cv2.imread(str(file_path))

                        if img is None:
                            continue

                        height, width = img.shape[:2]
                        label = file_path.parent.name
                        channels = img.shape[2] if len(img.shape) == 3 else 1


                        # print(file_path,label, height, width)


                        record = {
                            "file_path": str(file_path),
                            "label": label,
                            "height": height,
                            "width": width,
                            "channels": channels
                        }

                        self.records.append(record)
        self.dataframe = pd.DataFrame(self.records)
        return self.dataframe