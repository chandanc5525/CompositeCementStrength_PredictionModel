from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def get_models(random_state=42):

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(),
        "elastic_net": ElasticNet(),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "adaboost": AdaBoostRegressor(random_state=random_state),
        "extra_trees": ExtraTreesRegressor(random_state=random_state),
        "svr": SVR(),
        "knn": KNeighborsRegressor()
    }

    return models