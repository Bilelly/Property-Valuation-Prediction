from sklearn.model_selection import KFold, cross_val_score
import numpy as np

def cross_validate_model(model, X, y, cv=5, scoring="neg_mean_absolute_error"):
    """
    Perform K-Fold Cross Validation
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "all_scores": scores
    }
