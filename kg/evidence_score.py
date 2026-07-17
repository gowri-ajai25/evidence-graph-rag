"""
Computes a composite evidence score (0 to 1) for each drug-gene relationship.
Combines: number of supporting sources, CIViC evidence level, citation count,
and clinical DIRECTION (sensitizing vs. resistant) -- a relationship backed
by strong evidence of RESISTANCE should not score as a positive recommendation.
"""

EVIDENCE_LEVEL_WEIGHTS = {
    "A": 1.0,
    "B": 0.75,
    "C": 0.5,
    "D": 0.3,
    "E": 0.15,
}

# How clinical significance affects whether this relationship SUPPORTS
# using the drug, or argues AGAINST it.
RESISTANCE_TERMS = {"resistance", "poor outcome", "adverse response", "reduced sensitivity"}
SUPPORTIVE_TERMS = {"sensitivity/response", "sensitivity", "response", "better outcome", "positive", "favorable"}


def score_from_evidence_level(level):
    if level is None or (isinstance(level, float)):
        return 0.3
    level = str(level).strip().upper()
    return EVIDENCE_LEVEL_WEIGHTS.get(level, 0.3)


def score_from_source_count(num_sources, max_sources=3):
    return min(num_sources / max_sources, 1.0)


def direction_multiplier(clinical_significance):
    """
    Returns a multiplier applied to the evidence score based on clinical direction:
      +1.0  -> supportive evidence (sensitivity/response/better outcome) - full score
      -1.0  -> resistance/poor outcome evidence - score becomes negative (flag, not recommend)
       0.5  -> ambiguous/unlabeled categories (predisposition, oncogenicity) - reduced confidence
       0.3  -> unknown/missing - default low confidence, treated as neutral-positive
    """
    if clinical_significance is None or (isinstance(clinical_significance, float)):
        return 0.3

    sig = str(clinical_significance).strip().lower()

    if sig in RESISTANCE_TERMS:
        return -1.0
    if sig in SUPPORTIVE_TERMS:
        return 1.0
    # predisposition, oncogenicity, and other non-directional categories
    return 0.5


def composite_score(evidence_level=None, num_sources=1, has_fda_approval=False, clinical_significance=None):
    """
    Combines multiple evidence signals into one final score.
    Score range: -1.0 to 1.0
      Positive = evidence supports using this drug
      Negative = evidence suggests this drug would NOT work (resistance)

    Weighting before direction adjustment:
      - 50% evidence level (clinical maturity)
      - 30% number of independent sources
      - 20% FDA approval boost
    Then multiplied by the direction signal.
    """
    level_score = score_from_evidence_level(evidence_level)
    source_score = score_from_source_count(num_sources)
    fda_score = 1.0 if has_fda_approval else 0.0

    base_score = (
        0.5 * level_score +
        0.3 * source_score +
        0.2 * fda_score
    )

    direction = direction_multiplier(clinical_significance)
    final_score = base_score * direction

    return round(final_score, 3)


if __name__ == "__main__":
    print("BRCA1 -> Olaparib (Level A, FDA approved, 3 sources, Sensitivity):")
    print(composite_score(evidence_level="A", num_sources=3, has_fda_approval=True, clinical_significance="Sensitivity/Response"))

    print("\nTP53 -> Cisplatin (Level B, 1 source, RESISTANCE):")
    print(composite_score(evidence_level="B", num_sources=1, has_fda_approval=False, clinical_significance="Resistance"))

    print("\nUnknown gene -> experimental drug (no level, 1 source, no direction):")
    print(composite_score(evidence_level=None, num_sources=1, has_fda_approval=False, clinical_significance=None))