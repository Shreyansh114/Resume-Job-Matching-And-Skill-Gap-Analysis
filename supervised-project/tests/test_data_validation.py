import pandas as pd

from resume_matcher.data.validator import validate_pair_frame


def test_pair_validation_requires_text_and_binary_target():
    frame = pd.DataFrame(
        {
            "resume_text": ["Python developer", "SQL analyst"],
            "job_description": ["Python role", "SQL role"],
            "label": [1, 0],
        }
    )
    assert validate_pair_frame(frame, require_labels=True).is_valid


def test_pair_validation_rejects_missing_target_and_invalid_labels():
    missing_target = pd.DataFrame({"resume_text": ["text"], "job_description": ["job"]})
    assert not validate_pair_frame(missing_target, require_labels=True).is_valid

    invalid_label = missing_target.assign(label=[2])
    assert not validate_pair_frame(invalid_label, require_labels=True).is_valid
