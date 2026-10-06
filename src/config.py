from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR.parent / "datasets" / "weather"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DATASET_PATH = DATASET_DIR / "weatherAUS_2.csv"


TEST_SIZE = 0.20
RANDOM_STATE = 42

CV_SPLITS = 5
SCORING = "accuracy"

LOCATIONS = [
    "Melbourne",
    "MelbourneAirport",
    "Watsonia",
]

TARGET = "RainTomorrow"

FEATURES = [
    "Location",
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustDir",
    "WindGustSpeed",
    "WindDir9am",
    "WindDir3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday",
    "Month",
]

RANDOM_FOREST_PARAM_GRID = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
}

LOGISTIC_REGRESSION_PARAM_GRID = {
    "classifier__solver": ["liblinear"],
    "classifier__penalty": ["l1", "l2"],
    "classifier__class_weight": [None, "balanced"],
}