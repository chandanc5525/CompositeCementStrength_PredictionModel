import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from src.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_train, X_test, y_train, y_test, cv=False, folds=5):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results = {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }

    if cv:
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=folds,
            scoring="r2"
        )
        results["cv_mean_r2"] = cv_scores.mean()

    logger.info(f"Model evaluated: R2={r2:.4f}")

    return results