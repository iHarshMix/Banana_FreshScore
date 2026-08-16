"""Quality Gate logic comparing candidate model against production baseline."""

from typing import Tuple

from src.banana_mlops.utils.logger import setup_logger

logger = setup_logger("banana_mlops.models.evaluate_gate")


def evaluate_quality_gate(
    candidate_mae: float,
    production_mae: float,
    max_acceptable_mae: float = 0.08,
) -> Tuple[bool, str]:
    """Determine whether candidate model passes production quality gate.

    Quality Gate Rules:
    1. Candidate MAE must be strictly lower than active production MAE
       OR within target quality threshold.
    2. Candidate MAE must not exceed maximum acceptable threshold (0.08).

    Args:
        candidate_mae: Mean Absolute Error of newly trained candidate model.
        production_mae: Mean Absolute Error of current production model.
        max_acceptable_mae: Hard upper bound on acceptable error.

    Returns:
        Tuple of (passed: bool, rationale: str).
    """
    if candidate_mae > max_acceptable_mae:
        rationale = (
            f"REJECT: Candidate MAE ({candidate_mae:.4f}) exceeds hard ceiling "
            f"threshold of {max_acceptable_mae:.4f}."
        )
        logger.warning(rationale)
        return False, rationale

    if candidate_mae <= production_mae:
        rationale = (
            f"PROMOTE: Candidate MAE ({candidate_mae:.4f}) improves upon or matches "
            f"active production MAE ({production_mae:.4f})."
        )
        logger.info(rationale)
        return True, rationale

    # If candidate is slightly worse than production but still very healthy (< 0.05)
    if candidate_mae <= 0.05:
        rationale = (
            f"PROMOTE (Healthy): Candidate MAE ({candidate_mae:.4f}) meets high "
            f"quality standard (< 0.05)."
        )
        logger.info(rationale)
        return True, rationale

    rationale = (
        f"REJECT: Candidate MAE ({candidate_mae:.4f}) is worse than production MAE "
        f"({production_mae:.4f})."
    )
    logger.warning(rationale)
    return False, rationale
