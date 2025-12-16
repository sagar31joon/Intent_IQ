# intent_system/model_handlers.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier


class BaseModelHandler:
    """Common interface for all model training handlers."""
    
    def train(self, embeddings, labels):
        raise NotImplementedError("train() must be implemented by subclasses.")


class LogisticRegressionHandler(BaseModelHandler):
    def __init__(self):
        self.model = LogisticRegression(max_iter=2000)

    print("[TRAINER] Training Logistic Regression...")

    def train(self, embeddings, labels):
        self.model.fit(embeddings, labels)
        return self.model


class SVCHandler(BaseModelHandler):
    def __init__(self):
        self.model = SVC(kernel="rbf", probability=True)

    print("[TRAINER] Training SVC (RBF)...")

    def train(self, embeddings, labels):
        self.model.fit(embeddings, labels)
        return self.model


# Skeleton for later
class NeuralNetHandler(BaseModelHandler):
    def __init__(self):
        raise NotImplementedError("NeuralNet model not implemented yet.")
