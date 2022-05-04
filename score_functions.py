class Scorer:
    pass


class F1(Scorer):
    def __init__(self, name=None):
        self.name = name

    def __call__(self, TP, FP, TN, FN):
        try:
            return TP / (TP + FN)
        except ZeroDivisionError:
            print("Divided by 0")
            return 0


def prediction_scores(predictions, y):
    TP = (predictions == y and predictions == 1).sum()
    FP = (predictions != y and predictions == 1).sum()
    FN = (predictions != y and predictions == 0).sum()
    TN = (predictions == y and predictions == 0).sum()
    return TP, FP, TN, FN