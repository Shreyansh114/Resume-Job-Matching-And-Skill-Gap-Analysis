"""Group-aware random-forest hyperparameter search."""

from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold


def tune_random_forest(
    features,
    labels: Sequence[int],
    *,
    groups: Sequence[str] | None = None,
    cv_folds: int = 3,
    random_state: int = 42,
) -> GridSearchCV:
    """Fit a small F1-scored grid search, grouping folds when identifiers are supplied."""
    label_values = np.asarray(labels)
    if len(features) != len(label_values) or len(label_values) == 0:
        raise ValueError("Features and labels must have the same non-zero length.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if np.unique(label_values).size < 2:
        raise ValueError("Hyperparameter tuning requires both target classes.")

    if groups is not None:
        group_values = np.asarray(groups)
        if len(group_values) != len(label_values):
            raise ValueError("groups must have the same length as labels.")
        if np.unique(group_values).size < cv_folds:
            raise ValueError("There must be at least cv_folds unique groups.")
        cross_validation = GroupKFold(n_splits=cv_folds)
    else:
        smallest_class = int(np.unique(label_values, return_counts=True)[1].min())
        if smallest_class < cv_folds:
            raise ValueError("Each class must have at least cv_folds rows for stratified tuning.")
        cross_validation = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )

    search = GridSearchCV(
        estimator=RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        param_grid={
            "n_estimators": [100, 300],
            "max_depth": [4, 8, None],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        },
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
    )
    if groups is None:
        search.fit(features, label_values)
    else:
        search.fit(features, label_values, groups=np.asarray(groups))
    return search
