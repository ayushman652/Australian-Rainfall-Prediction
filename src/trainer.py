from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import  CV_SPLITS, LOGISTIC_REGRESSION_PARAM_GRID, RANDOM_FOREST_PARAM_GRID, SCORING

from .preprocessing import build_preprocessor


def train_model( model, param_grid, X_train, y_train ):
    """Train a model using preprocessing and GridSearchCV."""

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=CV_SPLITS,
        scoring=SCORING,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search