import os
from config import Config
from src.data_ingestion import load_data
from src.data_preprocessing import preprocess_data
from src.model_builder import get_models
from src.model_evaluator import evaluate_model
from src.model_predictor import save_model


def main():

    df, params = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df, params)

    models = get_models(params["project"]["random_state"])

    best_model = None
    best_score = -float("inf")
    best_name = None

    print("\nModel Performance:\n")

    for name, model in models.items():

        results = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            cv=params["training"]["cross_validation"],
            folds=params["training"]["cv_folds"]
        )

        print(f"{name} -> R2: {results['r2']:.4f}")

        if results["r2"] > best_score:
            best_score = results["r2"]
            best_model = model
            best_name = name

    os.makedirs(Config.MODELS_PATH, exist_ok=True)
    save_model(best_model, f"{Config.MODELS_PATH}/best_model.pkl")

    print("\nBest Model:", best_name)
    print("Best R2 Score:", round(best_score, 4))


if __name__ == "__main__":
    main()