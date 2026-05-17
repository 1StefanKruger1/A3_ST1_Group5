from src.services.dataset_indexer import DatasetIndexer
from src.services.image_preprocessor import ImagePreprocessor
from src.services.classifier_service import ClassifierService
from src.services.eda_service import EDAService
from src.config import EDA_OUTPUT_DIR
from src.config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS

class WorkflowService:
    def __init__(self):
        self.indexer = DatasetIndexer()
        self.preprocessor = ImagePreprocessor()
        self.dataframe = None
        self.classifier = ClassifierService(self.preprocessor)


    def load_dataframe(self):
        if self.dataframe is None:
            self.dataframe = self.indexer.build_dataframe()
        return self.dataframe

    def show_summary(self):
        ldf = self.load_dataframe()

        print("\n==================== Dataset Summary ====================")
        print("Total images:", len(ldf))
        print("Total classes:", ldf["label"].nunique())
        print("Average width:", round(ldf["width"].mean(), 2))
        print("Average height:", round(ldf["height"].mean(), 2))


    def count_classes(self):
        ldf = self.load_dataframe()
        print("\n ======================= Class counts =======================")
        print (ldf["label"].value_counts())


    def eda_analysis(self):
        ldf = self.load_dataframe()
        print("\n====================== EDA Analysis ======================")

        eda_service = EDAService(ldf, EDA_OUTPUT_DIR)
        outputs = eda_service.run_all()

        print("EDA files created successfully:")
        for name, path in outputs.items():
            print(f"{name}: {path}")

    def train_model(self):
        ldf = self.load_dataframe()

        print("\n====================== Train Model =======================")
        accuracy, report = self.classifier.train(ldf)
        print("Accuracy:", round(accuracy, 4))
        print("\nClassification Report")
        print("=====================")
        print(report)

    def predict_image(self):

        dataset_folder = RAW_DATA_DIR / "stream_macroinvertebrates"

        class_folders = [
            folder for folder in dataset_folder.iterdir()
            if folder.is_dir()
        ]

        if not class_folders:
            print("No class folders found.")
            return

        print("\nAvailable Classes")
        print("=================")

        for index, folder in enumerate(class_folders, start=1):
            print(f"{index}. {folder.name}")

        class_choice = input("Choose class number: ").strip()

        if not class_choice.isdigit():
            print("Invalid choice.")
            return

        class_index = int(class_choice) - 1

        if class_index < 0 or class_index >= len(class_folders):
            print("Class number is out of range.")
            return

        selected_class = class_folders[class_index]

        image_files = [
            file for file in selected_class.iterdir()
            if file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not image_files:
            print("No images found in this class.")
            return

        print("\nAvailable Images")
        print("================")

        for index, image_file in enumerate(image_files, start=1):
            print(f"{index}. {image_file.name}")

        image_choice = input("Choose image number: ").strip()

        if not image_choice.isdigit():
            print("Invalid choice.")
            return

        image_index = int(image_choice) - 1

        if image_index < 0 or image_index >= len(image_files):
            print("Image number is out of range.")
            return

        selected_image = image_files[image_index]

        prediction = self.classifier.predict(selected_image)

        if isinstance(prediction, tuple):
            predicted_class, confidence = prediction

            print("\nPrediction Result")
            print("=================")
            print("Selected image:", selected_image.name)
            print("Actual folder:", selected_class.name)
            print("Predicted class:", predicted_class)

            if confidence is not None:
                print("Confidence:", f"{confidence:.2%}")

        else:
            print("\nPrediction Result")
            print("=================")
            print("Selected image:", selected_image.name)
            print("Actual folder:", selected_class.name)
            print("Predicted class:", prediction)





