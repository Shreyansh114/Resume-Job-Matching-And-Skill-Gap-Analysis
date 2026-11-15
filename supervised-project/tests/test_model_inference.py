from types import SimpleNamespace

import numpy as np

from resume_matcher.models.evaluate import binary_metrics
from resume_matcher.models.explainability import forest_feature_importance


def test_binary_metrics_return_confusion_matrix_and_auc():
    metrics = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["f1"] == 0.8
    assert np.isclose(metrics["roc_auc"], 1.0)


def test_feature_importance_is_ranked_and_named():
    model = SimpleNamespace(feature_importances_=np.array([0.25, 0.75]))
    result = forest_feature_importance(model, ["text", "skills"])
    assert result[0] == {"feature": "skills", "importance": 0.75}
