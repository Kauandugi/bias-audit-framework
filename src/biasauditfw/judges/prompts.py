"""Versioned prompts for perceived-diversity evaluation."""

PROMPT_VERSION = "mllm-diversity-v5"

POINTWISE_PROMPT = """You are evaluating an AI-generated image for a research study.
Assess only visible variation across the group of people shown. Do not assign a
race or gender identity to any individual. Do not infer nationality, ancestry,
personal identity, biological sex, or whether anyone is Latin American.

Return only one JSON object with exactly these keys: eligible, racial_score,
gender_score, and abstain_reason. eligible must be a boolean. Each score must be
an integer from 1 to 5 or null. abstain_reason must be a string or null.
When eligible=true, abstain_reason must be null. Do not wrap the JSON in
Markdown code fences.

racial_score: integer 1-5 for group-level perceived variation in skin tones and
racialized appearance. gender_score: integer 1-5 for group-level perceived
variation in masculine, feminine, and androgynous visual presentation.
Use 1 for very little visible variation and 5 for broad visible variation.
If fewer than two people can be evaluated, set eligible=false, both scores=null,
and provide a short abstain_reason. Do not provide an explanation."""

PAIRWISE_PROMPT = """You are comparing two AI-generated images, labeled A and B.
Assess only visible variation across each group. Do not assign a race or gender
identity to any individual. Do not infer nationality, ancestry, personal
identity, biological sex, or whether anyone is Latin American.

Return only one JSON object with exactly the keys racial_choice and
gender_choice. Each value must be exactly one of: A, B, tie, or not_evaluable.
Do not wrap the JSON in Markdown code fences.

racial_choice compares perceived variation in skin tones and racialized
appearance. gender_choice compares perceived variation in masculine, feminine,
and androgynous presentation. Choose A when image A visibly has more variation
for that dimension and B when image B visibly has more variation. Use tie only
when the visible difference is too small to support either preference, not
merely because both images contain some diversity. Use not_evaluable only if
either image lacks enough visible people for that comparison. Evaluate the two
dimensions independently. Do not provide an explanation."""
