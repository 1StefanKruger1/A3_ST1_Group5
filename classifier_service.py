import joblib
import numpy as np
# import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.config import MODEL_OUTPUT_DIR


class ClassifierService:

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def prepare_features(self, dataframe):
        features = []
        labels = []

        for _, row in dataframe.iterrows():
            features.append(self.preprocessor.transform(row["file_path"]))
            labels.append(row["label"])

        return np.array(features), np.array(labels)

    def train(self, dataframe):
        x, y = self.prepare_features(dataframe)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        self.model.fit(x_train, y_train)

        predictions = self.model.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions,zero_division=0)

        report_path = MODEL_OUTPUT_DIR / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")

        self.save_model()

        return accuracy, report

    def save_model(self):
        model_path = MODEL_OUTPUT_DIR / "macro_classifier.joblib"
        joblib.dump(self.model, model_path)
        return model_path

    def load_model(self):
        model_path = MODEL_OUTPUT_DIR / "macro_classifier.joblib"

        if not model_path.exists():
            raise FileNotFoundError("Model not found. Please train the model first.")

        self.model = joblib.load(model_path)

    def predict(self, file_path):
        self.load_model()

        features = self.preprocessor.transform(file_path).reshape(1, -1)
        prediction = self.model.predict(features)[0]

        confidence =None
        if hasattr(self.model, "predict_proba"):
            confidence = self.model.predict_proba(features).max()

        return prediction, confidence
