# Data dictionary and preparation notes

## Training pair table

The supervised learner consumes one row per resume/job pair.

| Field | Type | Required | Notes |
|---|---|---:|---|
| `resume_text` | string | yes | Prefer de-identified, extracted resume content. |
| `job_description` | string | yes | Include the requirements and responsibilities used for matching. |
| `label` | integer 0/1 | either `label` or `match_score` | Gold label with a documented definition. |
| `match_score` | float 0–1 | either `label` or `match_score` | Converted to binary using the configured threshold. |
| `resume_id` | string | no | Group identifier used to keep candidates out of both splits. |
| `job_id` | string | no | Retain for analysis and stronger group/temporal split design. |
| `resume_years_experience` | number | no | Non-negative years; do not infer from protected or unrelated data. |
| `job_years_experience` | number | no | Required years when explicitly stated. |
| `resume_education` | string | no | Normalized education category or documented free text. |
| `job_education` | string | no | Normalized minimum education requirement. |

Rows with missing optional structured fields remain usable; availability indicators are generated. Missing required text or invalid target values fail validation.

## Source files

Source datasets often use different column names and may store resume/job records separately from labels. Keep originals unchanged in `data/raw/`. Create a documented, de-identified mapping into the pair-table contract before calling `train_model`. Do not assume identifiers or field names based on filenames.

## Feature matrix

`build_pair_features` emits stable numeric columns: TF-IDF or configured embedding similarity, skill Jaccard/overlap/Dice, required/matched/missing skill counts, experience gap and availability, and exact education match and availability. The current taxonomy is only a curated starter list; domain experts should review, version, and expand canonical skills and aliases.

Never put names, email addresses, phone numbers, photos, age, gender, race, disability, or other sensitive/protected attributes into labels or features. Direct contact details are redacted during text normalization, but this is not a full anonymization guarantee.
