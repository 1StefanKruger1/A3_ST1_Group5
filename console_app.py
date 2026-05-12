from src.services.dataset_indexer import DatasetIndexer


class ConsoleApp:

    def __init__(self):
        self.indexer = DatasetIndexer()
        self.dataframe = None

    def run(self):

        while True:

            print("\nMacroinvertebrate Image Analysis System")
            print("1. Build Dataset Index")
            print("2. Show Dataset Summary")
            print("3. Show Class Counts")
            print("4.Export Dataset CSV ")
            print("5.Exit ")

            choice = input("Choose an option: ")

            if choice == "1":
                self.build_index()

            elif choice == "2":
                self.show_summary()

            elif choice == "3":
                self.show_class_counts()

            elif choice == "4":
                self.export_csv()

            elif choice == "5":
                print("Goodbye.")
                break

            else:
                print("Invalid option.")

    def build_index(self):

        self.dataframe = self.indexer.build_dataframe()

        if self.dataframe.empty:
            print("No images found.")

        else:
            print("Dataset indexed successfully.")
            print(f"Total images: {len(self.dataframe)}")

    def show_summary(self):

        if self.dataframe is None:
            self.build_index()

        if self.dataframe.empty:
            return

        print("\nDATASET SUMMARY")
        print(f"Total Images: {len(self.dataframe)}")
        print(f"Total Classes: {self.dataframe['label'].nunique()}")
        print(f"Average Width: {self.dataframe['width'].mean():.2f}")
        print(f"Average Height: {self.dataframe['height'].mean():.2f}")

    def show_class_counts(self):

        if self.dataframe is None:
            self.build_index()

        if self.dataframe.empty:
            return

        print("\nCLASS COUNTS")
        print(self.dataframe["label"].value_counts())

    def export_csv(self):

            if self.dataframe is None:
                self.build_index()

            if self.dataframe.empty:
                return

            output_path = "../outputs/dataset_index.csv" 
            self.dataframe.to_csv(output_path, index=False)

            print(f"\nCSV exported successfully to: {output_path}")


def main():

    app = ConsoleApp()
    app.run()


if __name__ == "__main__":
    main()