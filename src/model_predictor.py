import joblib


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def predict(model, X):
    return model.predict(X)