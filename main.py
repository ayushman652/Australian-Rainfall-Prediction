from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import LOCATIONS, LOGISTIC_REGRESSION_PARAM_GRID, RANDOM_FOREST_PARAM_GRID, RANDOM_STATE
from src.data_loader import load_data
from src.feature_engineering import add_date_features
from src.preprocessing import split_data
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.visualizer import plot_confusion_matrix, plot_feature_importance

def main():
    # Load dataset
    dataframe = load_data()

    # Keep only selected locations
    dataframe = dataframe[dataframe["Location"].isin(LOCATIONS)]

    # Create date-based features
    dataframe = add_date_features(dataframe)

    # Split features and target
    X_train, X_test, y_train, y_test = split_data(dataframe)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training features:", X_train.shape[1])

    logistic_model = train_model(
        LogisticRegression(),
        LOGISTIC_REGRESSION_PARAM_GRID,
        X_train,
        y_train,
    )
    
    random_forest_model = train_model(
        RandomForestClassifier(random_state=RANDOM_STATE),
        RANDOM_FOREST_PARAM_GRID,
        X_train,
        y_train,
    )
    
    print("Best Logistic Regression parameters:")
    print(logistic_model.best_params_)

    print("\nBest Random Forest parameters:")
    print(random_forest_model.best_params_)
    
    
    logistic_accuracy, logistic_report, logistic_matrix = evaluate_model(
    logistic_model,
    X_test,
    y_test,
    )

    random_forest_accuracy, random_forest_report, random_forest_matrix = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
    )
    
    print("\n--- Logistic Regression ---")
    print("Accuracy:", logistic_accuracy)
    print(logistic_report)
    print("Confusion Matrix:")
    print(logistic_matrix)

    print("\n--- Random Forest ---")
    print("Accuracy:", random_forest_accuracy)
    print(random_forest_report)
    print("Confusion Matrix:")
    print(random_forest_matrix) 
    
    
    # Visualizations
    plot_confusion_matrix(
        logistic_model,
        "logistic_regression",
        X_test,
        y_test,
        "Logistic Regression - Confusion Matrix",
    )

    plot_confusion_matrix(
        random_forest_model,
        "random_forest",
        X_test,
        y_test,
        "Random Forest - Confusion Matrix",
    )

    plot_feature_importance(
        random_forest_model,
        "Random Forest - Feature Importance",
    )

if __name__ == "__main__":
    main()