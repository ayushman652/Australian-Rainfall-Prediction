import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from .config import OUTPUT_DIR


def plot_confusion_matrix(model,model_name, X_test, y_test, title):
    """Plot the confusion matrix of a trained model."""
    predictions = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        cmap="Blues",
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(
            OUTPUT_DIR / f"{model_name.lower()}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
    
    plt.show()


def plot_feature_importance(model, title, top_n=15):
    """Plot the most important features of a Random Forest model."""
    classifier = model.best_estimator_.named_steps["classifier"]
    preprocessor = model.best_estimator_.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_data = sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    features = [item[0] for item in importance_data]
    values = [item[1] for item in importance_data]

    plt.figure(figsize=(10, 6))
    plt.barh(features[::-1], values[::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(
            OUTPUT_DIR / "random_forest_feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
    
    plt.show()