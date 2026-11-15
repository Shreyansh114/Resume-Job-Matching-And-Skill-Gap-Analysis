# Research notebooks

Keep notebooks focused on reproducible exploration and record dataset versions, preprocessing choices, and evaluation split strategy. Do not commit raw resume text or notebook outputs containing personal information. Move validated transformations into `src/resume_matcher/` so training and dashboard inference share the same implementation.

Suggested sequence:

1. Explore schema, missingness, label distribution, duplicate pairs, and identifiers.
2. Audit taxonomy coverage and compare lexical/embedding feature baselines.
3. Evaluate group-safe splits, threshold trade-offs, subgroup metrics, and error cases.
