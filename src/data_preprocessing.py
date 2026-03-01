from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from config import Config
from src.logger import get_logger

logger = get_logger(__name__)


def preprocess_data(df, params):

    target_col = params["data"]["target_column"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["split"]["test_size"],
        shuffle=params["split"]["shuffle"],
        random_state=params["project"]["random_state"]
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    os.makedirs(Config.MODELS_PATH, exist_ok=True)
    joblib.dump(scaler, f"{Config.MODELS_PATH}/scaler.pkl")

    logger.info("Data preprocessing completed")

    return X_train, X_test, y_train, y_test