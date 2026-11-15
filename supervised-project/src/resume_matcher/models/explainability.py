"""Model-level feature importance summaries."""

import pandas as pd


def forest_feature_importance(model, feature_names: list[str]) -> list[dict[str, float | str]]:
    """Return sorted impurity-based importance for a fitted tree ensemble."""
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        raise ValueError("The fitted model does not expose feature_importances_.")
    if len(importance) != len(feature_names):
        raise ValueError("Feature names do not match the model's fitted feature count.")
    ranking = pd.DataFrame({"feature": feature_names, "importance": importance})
    ranking = ranking.sort_values("importance", ascending=False, kind="stable")
    return [
        {"feature": str(row.feature), "importance": float(row.importance)}
        for row in ranking.itertuples(index=False)
    ]
