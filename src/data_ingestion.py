import pandas as pd
import yaml
from src.logger import get_logger

logger = get_logger(__name__)


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data():
    params = load_params()
    data_path = params["data"]["raw_path"]

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    logger.info(f"Data shape: {df.shape}")
    return df, params