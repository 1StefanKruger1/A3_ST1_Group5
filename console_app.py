from src.services.workflow_service import WorkflowService

# from src.services

class ConsoleApp:
    def __init__(self):
        self.service = WorkflowService()

    def run(self):
        while True:
            print("_______________________________________________")
            print("\n Image Analysis System of Macroinvertebrate ")
            print("_______________________________________________")
            print("1. Show dataset summary")
            print("2. Show class counts")
            print("3. Build EDA ")
            print("4. Train classifier model")
            print("5. Predict an image")
            print("6. Exit")

            choice = int(input("Choose one of the options from 1 to 6: ").strip())

            try:
                if choice == 1:
                    self.service.show_summary()
                elif choice == 2:
                    self.service.count_classes()
                elif choice == 3:
                    self.service.eda_analysis()
                elif choice == 4:
                    self.service.train_model()
                elif choice == 5:
                    self.service.predict_image()
                elif choice == 6:
                    print("Program Closing")
                    break

                else:
                    print("Unknown option. Please choose numbers between 1 and 6.")

            except ValueError:
                print("Invalid option. Please choose the options between 1 and 6.")


