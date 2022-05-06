from abc import ABC, abstractmethod


class Scorer(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, TP, FP, TN, FN):
        pass


class F1(Scorer):
    def __init__(self):
        super().__init__("F1")

    def __call__(self, TP, FP, TN, FN):
        print(f"{TP=}, {FP=}, {TN=}, {FN=}")
        try:
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            return 2 / (1 / recall + 1 / precision)
        except ZeroDivisionError:
            print(f"Divided by 0. {TP=}, {FP=}, {TN=}, {FN=}")
            return 0


def prediction_scores(predictions, y):
    TP = ((predictions == y) & (predictions == 1)).sum()
    FP = ((predictions != y) & (predictions == 1)).sum()
    FN = ((predictions != y) & (predictions == 0)).sum()
    TN = ((predictions == y) & (predictions == 0)).sum()
    return TP, FP, TN, FN
