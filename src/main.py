from training.hyperparameter_search import HyperparameterSearch


class Main:
    def __init__(self):
        self.hyperparameter_search = HyperparameterSearch()

    def run_search(self):
        self.hyperparameter_search.search_hyperparameters()


if __name__ == "__main__":
    main = Main()
    main.run_search()
