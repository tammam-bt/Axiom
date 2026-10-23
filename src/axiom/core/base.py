class BaseEstimator:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        raise NotImplementedError("The fit method must be implemented by the subclass.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("The predict method must be implemented by the subclass.")