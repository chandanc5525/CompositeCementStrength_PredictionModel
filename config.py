class Config:
    PROJECT_NAME = "CementStrength Prediction Model"
    VERSION = "1.0.0"

    # ========================
    # Paths
    # ========================
    DATA_FILENAME = "Concrete_Data.csv"
    DATA_RAW_PATH = "data/raw"
    DATA_PROCESSED_PATH = "data/processed"
    MODELS_PATH = "models"
    LOGS_PATH = "logs"
    RESEARCH_PATH = "research"
    ARTIFACTS_PATH = "artifacts"

    # ========================
    # Target Column
    # ========================
    TARGET_COLUMN = "Concrete compressive strength(MPa, megapascals) "

    # ========================
    # Data Split
    # ========================
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1

    # ========================
    # Model Parameters
    # ========================
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = None

    # ========================
    # MLflow Settings
    # ========================
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME = PROJECT_NAME

    # ========================
    # API Settings (FastAPI)
    # ========================
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    DEBUG = True

    # ========================
    # DVC Settings
    # ========================
    DVC_REMOTE = "remote_storage"

    # ========================
    # Logging
    # ========================
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"