## ANN Architecture for Model Design : Tensorflow 
---

```
# ==========================================================
# 1. IMPORTS & REPRODUCIBILITY
# ==========================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import joblib
import logging

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(level=logging.INFO)

# ==========================================================
# 2. DATA SPLIT
# ==========================================================

def split_data(data, target_col, test_size=0.3, random_state=42):

    X = data.drop(columns=[target_col])
    y = data[target_col]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

# ==========================================================
# 3. PREPROCESSING
# ==========================================================

def preprocess_data(X_train, X_test):

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor

# ==========================================================
# 4. ANN MODEL BUILDER
# ==========================================================

def build_ann(input_dim,
              hidden_layers=[128, 64, 32],
              dropout_rate=0.3,
              learning_rate=0.001):

    model = Sequential()

    for i, units in enumerate(hidden_layers):

        if i == 0:
            model.add(Dense(units, activation="relu",
                            input_shape=(input_dim,)))
        else:
            model.add(Dense(units, activation="relu"))

        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation="linear"))  # Regression output

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return model

# ==========================================================
# 5. K-FOLD CROSS VALIDATION (TRAIN ONLY)
# ==========================================================

def kfold_ann_cv(X_train, y_train, input_dim, folds=5):

    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)

    rmse_scores = []
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):

        print(f"\nFold {fold}")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = build_ann(input_dim=input_dim)

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )

        y_pred = model.predict(X_val).ravel()

        rmse_scores.append(
            np.sqrt(mean_squared_error(y_val, y_pred))
        )
        r2_scores.append(
            r2_score(y_val, y_pred)
        )

        print(f"RMSE: {rmse_scores[-1]:.4f}")
        print(f"R2  : {r2_scores[-1]:.4f}")

    print("\nK-Fold Mean Performance")
    print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Mean R2  : {np.mean(r2_scores):.4f}")

    return np.mean(rmse_scores), np.mean(r2_scores)

# ==========================================================
# 6. HYPERPARAMETER TUNING (VALIDATION BASED)
# ==========================================================

def ann_hyperparameter_tuning(X_train, y_train, input_dim):

    param_grid = {
        "hidden_layers": [
            [128, 64],
            [128, 64, 32],
            [256, 128, 64]
        ],
        "dropout_rate": [0.2, 0.3],
        "learning_rate": [0.001, 0.0005]
    }

    best_val_r2 = -np.inf
    best_model = None
    best_config = None

    for layers in param_grid["hidden_layers"]:
        for dropout in param_grid["dropout_rate"]:
            for lr in param_grid["learning_rate"]:

                print("\nTesting Configuration")
                print(layers, dropout, lr)

                model = build_ann(
                    input_dim=input_dim,
                    hidden_layers=layers,
                    dropout_rate=dropout,
                    learning_rate=lr
                )

                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True
                )

                history = model.fit(
                    X_train,
                    y_train,
                    validation_split=0.2,
                    epochs=200,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )

                val_r2 = max(history.history.get("val_mae", [0]))

                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_model = model
                    best_config = (layers, dropout, lr)

    print("\nBest Configuration Found")
    print(best_config)

    return best_model, best_config

# ==========================================================
# 7. FINAL EVALUATION
# ==========================================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test).ravel()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return rmse, mae, r2

# ==========================================================
# 8. MAIN EXECUTION
# ==========================================================

TARGET_COL = "heart_disease_risk_score"  # change if needed

def main():

    logging.info("ANN Pipeline Started")

    # Step 1: Split
    X_train, X_test, y_train, y_test = split_data(
        df, TARGET_COL
    )

    # Step 2: Preprocess
    X_train_processed, X_test_processed, preprocessor = preprocess_data(
        X_train, X_test
    )

    # Step 3: K-Fold CV
    kfold_ann_cv(
        X_train_processed,
        y_train,
        input_dim=X_train_processed.shape[1]
    )

    # Step 4: Hyperparameter Tuning
    best_model, best_config = ann_hyperparameter_tuning(
        X_train_processed,
        y_train,
        input_dim=X_train_processed.shape[1]
    )

    # Step 5: Final Test Evaluation
    rmse, mae, r2 = evaluate_model(
        best_model,
        X_test_processed,
        y_test
    )

    print("\nFinal Test Performance")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")

    model.save("final_ann_model.keras")
    joblib.dump(preprocessor, "preprocessor.pkl")

    logging.info("ANN Pipeline Completed")

    return best_model

if __name__ == "__main__":
    main()

```
---

## ANN Acrchitecture for Model Design : Pytorch

```

# ==========================================================
# 1. IMPORTS & REPRODUCIBILITY
# ==========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import joblib
import logging

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(level=logging.INFO)

# ==========================================================
# 2. DATA SPLIT
# ==========================================================

def split_data(data, target_col, test_size=0.3, random_state=42):

    X = data.drop(columns=[target_col])
    y = data[target_col]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

# ==========================================================
# 3. PREPROCESSING (TRAIN ONLY FIT)
# ==========================================================

def preprocess_data(X_train, X_test):

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor

# ==========================================================
# 4. PYTORCH ANN MODEL
# ==========================================================

class ANNModel(nn.Module):

    def __init__(self, input_dim, hidden_layers=[128, 64, 32], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = units

        layers.append(nn.Linear(prev_dim, 1))  

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ==========================================================
# 5. TRAINING FUNCTION
# ==========================================================

def train_model(model, X_train, y_train,
                X_val, y_val,
                epochs=200,
                lr=0.001):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    return model

# ==========================================================
# 6. K-FOLD CROSS VALIDATION (TRAIN ONLY)
# ==========================================================

def kfold_cv(X_train_np, y_train_np, input_dim, folds=5):

    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)

    rmse_scores = []
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np), 1):

        print(f"\nFold {fold}")

        X_tr = torch.tensor(X_train_np[train_idx], dtype=torch.float32).to(device)
        y_tr = torch.tensor(y_train_np[train_idx], dtype=torch.float32).view(-1,1).to(device)

        X_val = torch.tensor(X_train_np[val_idx], dtype=torch.float32).to(device)
        y_val = torch.tensor(y_train_np[val_idx], dtype=torch.float32).view(-1,1).to(device)

        model = ANNModel(input_dim).to(device)

        model = train_model(model, X_tr, y_tr, X_val, y_val)

        model.eval()
        with torch.no_grad():
            preds = model(X_val).cpu().numpy().ravel()

        rmse = np.sqrt(mean_squared_error(y_train_np[val_idx], preds))
        r2 = r2_score(y_train_np[val_idx], preds)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"RMSE: {rmse:.4f}")
        print(f"R2  : {r2:.4f}")

    print("\nMean CV Performance")
    print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Mean R2  : {np.mean(r2_scores):.4f}")

    return np.mean(rmse_scores), np.mean(r2_scores)

# ==========================================================
# 7. HYPERPARAMETER TUNING
# ==========================================================

def hyperparameter_tuning(X_train_np, y_train_np, input_dim):

    hidden_options = [
        [128, 64],
        [128, 64, 32],
        [256, 128, 64]
    ]

    dropout_options = [0.2, 0.3]
    lr_options = [0.001, 0.0005]

    best_r2 = -np.inf
    best_model = None
    best_config = None

    for hidden in hidden_options:
        for dropout in dropout_options:
            for lr in lr_options:

                print("\nTesting:", hidden, dropout, lr)

                X_tr = torch.tensor(X_train_np, dtype=torch.float32).to(device)
                y_tr = torch.tensor(y_train_np, dtype=torch.float32).view(-1,1).to(device)

                # simple 80-20 split for validation
                split = int(0.8 * len(X_tr))

                model = ANNModel(input_dim, hidden, dropout).to(device)

                model = train_model(
                    model,
                    X_tr[:split], y_tr[:split],
                    X_tr[split:], y_tr[split:],
                    lr=lr
                )

                model.eval()
                with torch.no_grad():
                    preds = model(X_tr[split:]).cpu().numpy().ravel()

                val_r2 = r2_score(
                    y_train_np[split:], preds
                )

                if val_r2 > best_r2:
                    best_r2 = val_r2
                    best_model = model
                    best_config = (hidden, dropout, lr)

    print("\nBest Configuration:", best_config)
    print("Best Validation R2:", best_r2)

    return best_model

# ==========================================================
# 8. MAIN EXECUTION
# ==========================================================

TARGET_COL = "heart_disease_risk_score"

def main():

    logging.info("PyTorch ANN Pipeline Started")

    # Step 1: Split
    X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)

    # Step 2: Preprocess
    X_train_processed, X_test_processed, preprocessor = preprocess_data(
        X_train, X_test
    )

    # Convert to numpy
    X_train_np = np.array(X_train_processed)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test_processed)
    y_test_np = np.array(y_test)

    # Step 3: K-Fold CV
    kfold_cv(X_train_np, y_train_np, input_dim=X_train_np.shape[1])

    # Step 4: Hyperparameter Tuning
    best_model = hyperparameter_tuning(
        X_train_np,
        y_train_np,
        input_dim=X_train_np.shape[1]
    )

    # Step 5: Final Test Evaluation
    best_model.eval()
    with torch.no_grad():
        test_preds = best_model(
            torch.tensor(X_test_np, dtype=torch.float32).to(device)
        ).cpu().numpy().ravel()

    rmse = np.sqrt(mean_squared_error(y_test_np, test_preds))
    mae = mean_absolute_error(y_test_np, test_preds)
    r2 = r2_score(y_test_np, test_preds)

    print("\nFinal Test Performance")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")

    torch.save(best_model.state_dict(), "final_pytorch_ann.pth")
    joblib.dump(preprocessor, "preprocessor.pkl")

    logging.info("Pipeline Completed")

    return best_model

if __name__ == "__main__":
    main()

```
---

## Machine Learning Architecture for Model Design

```
# Importing Data Manipulation Libraries
import pandas as pd
import numpy as np
# Import Data Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Import Filter Warning Libraries
import warnings
warnings.filterwarnings('ignore')
# Import Logging
import logging
logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s',
                    filemode = 'w',
                    filename = 'model.log',force = True)
# Import Scikit Learn Libraries for Machine Learning Model Building
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,learning_curve,KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost
from xgboost import XGBRegressor

# Multicolinearity test and treatment libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from collections import OrderedDict

import logging
def data_ingestion(data_source: str) -> pd.DataFrame:

    logging.info("Data Ingestion Started...")
    df = pd.read_csv(data_source)
    logging.info("Data Ingestion Completed Successfully")
    return df

def data_exploration(df: pd.DataFrame) -> pd.DataFrame:

    stats = []

    numerical_cols = df.select_dtypes(exclude='object').columns

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        LW = Q1 - 1.5 * IQR
        UW = Q3 + 1.5 * IQR

        outlier_flag = "Has Outliers" if df[(df[col] < LW) | (df[col] > UW)].shape[0] > 0 else "No Outliers"

        numerical_stats = OrderedDict({
            "Feature": col,
            "Minimum": df[col].min(),
            "Maximum": df[col].max(),
            "Mean": df[col].mean(),
            "Median": df[col].median(),
            "Mode": df[col].mode().iloc[0] if not df[col].mode().empty else np.nan,
            "25%": Q1,
            "75%": Q3,
            "IQR": IQR,
            "Standard Deviation": df[col].std(),
            "Skewness": df[col].skew(),
            "Kurtosis": df[col].kurt(),
            "Outlier Comment": outlier_flag
        })

        stats.append(numerical_stats)

    report = pd.DataFrame(stats)
    return report

def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include='object').columns

    summary = []
    for col in cat_cols:
        summary.append({
            "Feature": col,
            "Unique Values": df[col].nunique(),
            "Most Frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
            "Missing Values": df[col].isna().sum()
        })

    return pd.DataFrame(summary)

def split_data(data, target_col, test_size=0.3, random_state=42):
    X = data.drop(columns=[target_col])
    y = data[target_col]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def encode_categorical(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    cat_cols = X_train.select_dtypes(include="object").columns

    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()

        # Fit ONLY on train
        X_train[col] = le.fit_transform(X_train[col])

        # Transform test using same mapping
        X_test[col] = X_test[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

        encoders[col] = le

    return X_train, X_test, encoders

def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return rmse, r2
def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "Decision Tree": DecisionTreeRegressor(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boost": GradientBoostingRegressor(),
        "Ada Boost": AdaBoostRegressor(),
        "XG Boost": XGBRegressor()
    }

    results = []

    for name, model in models.items():
        rmse, r2 = train_evaluate_model(
            model, X_train, X_test, y_train, y_test
        )
        results.append([name, rmse, r2])

    return pd.DataFrame(
        results, columns=["Model Name", "RMSE", "R2 Score"]).sort_values("R2 Score", ascending=False)

def k_fold_cv(X_train, y_train, folds=10):
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "Decision Tree": DecisionTreeRegressor(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boost": GradientBoostingRegressor(),
        "Ada Boost": AdaBoostRegressor(),
        "XG Boost": XGBRegressor()
    }

    results = []

    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=folds, scoring="r2"
        )
        results.append([name, scores.mean(), scores.std()])

    return pd.DataFrame(
        results, columns=["Model Name", "CV Mean R2", "CV STD"]).sort_values("CV Mean R2", ascending=False)

def hyperparameter_tuning(X_train, y_train, folds=5):
    tuning_config = {
        "XGBoost": {
            "model": XGBRegressor(),
            "params": {
                "eta": [0.1, 0.2, 0.3],
                "max_depth": [3, 5, 7],
                "gamma": [0, 10, 20],
                "reg_lambda": [0, 1]
            }
        },
        "Random Forest": {
            "model": RandomForestRegressor(),
            "params": {
                "max_depth": [5, 10, 15],
                "max_features": ["sqrt", "log2", 3, 4]
            }
        }
    }

    best_models = {}

    for name, cfg in tuning_config.items():
        grid = GridSearchCV(
            cfg["model"],
            cfg["params"],
            cv=folds,
            scoring="r2",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_

    return best_models

def post_tuning_cv(best_models, X_train, y_train, folds=5):
    results = []

    for name, model in best_models.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=folds, scoring="r2"
        )
        results.append([name, scores.mean(), scores.std()])

    return pd.DataFrame(
        results, columns=["Model Name", "CV Mean R2", "CV STD"]).sort_values("CV Mean R2", ascending=False)

def final_test_evaluation(best_model, X_train, X_test, y_train, y_test):
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return rmse, r2

import logging

DATA_URL = "https://raw.githubusercontent.com/chandanc5525/CardioVascularRisk_AssessmentModel/refs/heads/main/data/raw/cardiovascular_risk_dataset.csv"
TARGET_COL = "heart_disease_risk_score"

def main():
    logging.info("ML Pipeline Started")

    # --------------------------------
    # Step 1: Data Ingestion
    # --------------------------------
    df = data_ingestion(DATA_URL)

    # --------------------------------
    # Step 2: Data Exploration (EDA)
    # --------------------------------
    numerical_report = data_exploration(df)
    categorical_report = categorical_summary(df)

    print("\nNumerical EDA Report:")
    print(numerical_report)

    print("\nCategorical Summary:")
    print(categorical_report)

    # --------------------------------
    # Step 3: Train–Test Split (ONCE)
    # --------------------------------
    X_train, X_test, y_train, y_test = split_data(
        data=df,
        target_col=TARGET_COL,
        test_size=0.3,
        random_state=42
    )

    logging.info("Train–Test split completed")

    # --------------------------------
    # Step 4: Baseline Model Comparison
    # --------------------------------

    X_train, X_test, encoders = encode_categorical(X_train, X_test)

    baseline_results = compare_models(
        X_train, X_test, y_train, y_test
    )

    print("\nBaseline Model Comparison:")
    print(baseline_results)

    # --------------------------------
    # Step 5: Cross Validation (TRAIN ONLY)
    # --------------------------------
    cv_results = k_fold_cv(
        X_train, y_train, folds=10
    )

    print("\nCross Validation Results (Before Tuning):")
    print(cv_results)

    # --------------------------------
    # Step 6: Hyperparameter Tuning
    # (TRAIN ONLY)
    # --------------------------------
    best_models = hyperparameter_tuning(
        X_train, y_train, folds=5
    )

    logging.info("Hyperparameter tuning completed")

    # --------------------------------
    # Step 7: Post-Tuning Cross Validation
    # --------------------------------
    post_cv_results = post_tuning_cv(
        best_models, X_train, y_train, folds=5
    )

    print("\nCross Validation Results (After Tuning):")
    print(post_cv_results)

    # --------------------------------
    # Step 8: Final Test Evaluation
    # (TEST USED ONLY ONCE)
    # --------------------------------
    best_model_name = post_cv_results.iloc[0]["Model Name"]
    best_model = best_models[best_model_name]

    final_rmse, final_r2 = final_test_evaluation(
        best_model, X_train, X_test, y_train, y_test
    )

    print("\nFinal Test Performance:")
    print(f"Best Model : {best_model_name}")
    print(f"RMSE       : {final_rmse}")
    print(f"R2 Score   : {final_r2}")

    logging.info("ML Pipeline Completed Successfully")

    return best_model

if __name__ == "__main__":
    main()

```