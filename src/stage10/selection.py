"""Stage 10 - Layer 6: Champion Model Selection.

Accepts a Layer 5 EvaluationReport and deterministically selects exactly
one champion model. Performs no evaluation, retraining, or persistence
-- purely a ranking/decision layer over metrics Layer 5 already computed.

Selection ranking (as specified for this layer):
    1. Lowest RMSE
    2. If tied, lowest MAE
    3. If tied, highest R2
    4. If still tied, preserve insertion order (first-inserted wins)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from src.stage10.common import ExecutionMode
from src.stage10.evaluation import EvaluationReport, ModelEvaluation
from src.stage10.exceptions import SelectionError
from src.stage10.logging_utils import log_call

logger = logging.getLogger("stage10")


@dataclass(frozen=True)
class RankedCandidate:
    """A single candidate's position in the full ranking.

    Attributes
    ----------
    rank:
        1-indexed rank; 1 is best.
    name, rmse, mae, r2:
        The candidate's identity and validation metrics.
    """

    rank: int
    name: str
    rmse: float
    mae: float
    r2: float


@dataclass(frozen=True)
class SelectionResult:
    """The selected champion and why it won.

    Attributes
    ----------
    champion_name:
        Name of the selected model.
    champion_evaluation:
        The full ModelEvaluation for the champion, carried through
        unchanged from Layer 5.
    tied_with:
        Names of other candidates whose (rmse, mae, r2) exactly matched
        the champion's, before the insertion-order tiebreak was applied.
        Empty if the champion won outright. Kept for transparency --
        without this, a tie resolved by insertion order would look
        indistinguishable from a clean win when the report is read
        later.
    selected_at:
        UTC timestamp recorded when selection completed.
    """

    champion_name: str
    champion_evaluation: ModelEvaluation
    tied_with: list[str]
    selected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class SelectionReport:
    """Aggregate result of Layer 6 selection.

    Attributes
    ----------
    champion:
        The winning SelectionResult.
    ranking:
        Every considered candidate, best to worst, rank 1..N.
    excluded_models:
        Name -> reason, for models that could not be considered (only
        ever populated in BEST_EFFORT mode, carried forward from
        EvaluationReport.failed_models).
    mode:
        The ExecutionMode this selection run used.
    total_duration_seconds:
        Wall-clock time for the entire select_champion_model call.
    selected_at:
        UTC timestamp recorded when the selection run completed.
    """

    champion: SelectionResult
    ranking: list[RankedCandidate]
    excluded_models: dict[str, str]
    mode: ExecutionMode
    total_duration_seconds: float
    selected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _validate_candidate_metrics(name: str, evaluation: ModelEvaluation) -> None:
    """Defensively re-check that a candidate's metrics are finite.

    Layer 5 already guards against non-finite predictions, but Layer 6
    does not assume that guarantee holds by the time it receives the
    data -- re-validating here fails fast with a clear message instead
    of silently mis-ranking a NaN as though it were meaningful.

    Raises
    ------
    SelectionError
        If rmse, mae, or r2 is NaN or infinite.
    """
    for metric_name, value in (
        ("rmse", evaluation.rmse),
        ("mae", evaluation.mae),
        ("r2", evaluation.r2),
    ):
        if not np.isfinite(value):
            raise SelectionError(
                f"Candidate '{name}' has a non-finite {metric_name} value "
                f"({value}); cannot be ranked."
            )


@log_call
def rank_candidates(results: dict[str, ModelEvaluation]) -> list[RankedCandidate]:
    """Rank every candidate by (rmse asc, mae asc, r2 desc, insertion order asc).

    Parameters
    ----------
    results:
        Candidate name -> ModelEvaluation, in insertion order (as
        produced by EvaluationReport.results).

    Returns
    -------
    A list of RankedCandidate, best (rank=1) to worst.

    Raises
    ------
    SelectionError
        If results is empty, or any candidate has a non-finite metric.
    """
    if not results:
        raise SelectionError("Cannot rank candidates: results is empty.")

    for name, evaluation in results.items():
        _validate_candidate_metrics(name, evaluation)

    indexed_items = list(
        enumerate(results.items())
    )  # (insertion_index, (name, evaluation))

    def sort_key(
        item: tuple[int, tuple[str, ModelEvaluation]],
    ) -> tuple[float, float, float, int]:
        insertion_index, (_, evaluation) = item
        return (evaluation.rmse, evaluation.mae, -evaluation.r2, insertion_index)

    ordered = sorted(indexed_items, key=sort_key)

    ranking = [
        RankedCandidate(
            rank=rank,
            name=name,
            rmse=evaluation.rmse,
            mae=evaluation.mae,
            r2=evaluation.r2,
        )
        for rank, (_, (name, evaluation)) in enumerate(ordered, start=1)
    ]

    logger.debug(
        "Ranked %d candidate(s); top: %s",
        len(ranking),
        ranking[0].name if ranking else None,
    )
    return ranking


@log_call
def select_champion_model(
    evaluation_report: EvaluationReport,
    mode: ExecutionMode = ExecutionMode.STRICT,
) -> SelectionReport:
    """Select exactly one champion model from a Layer 5 EvaluationReport.

    Parameters
    ----------
    evaluation_report:
        Layer 5 output. Only evaluation_report.results is ranked;
        evaluation_report.failed_models is carried into
        SelectionReport.excluded_models rather than being ranked.
    mode:
        STRICT (default): raise immediately if evaluation_report.failed_models
        is non-empty -- selection refuses to crown a champion from an
        incomplete field of candidates.
        BEST_EFFORT: proceed using only the successfully evaluated
        candidates; failed ones are recorded in excluded_models.

    Returns
    -------
    SelectionReport

    Raises
    ------
    SelectionError
        If evaluation_report.results is empty, if mode is STRICT and
        evaluation_report.failed_models is non-empty, or if any
        candidate has a non-finite metric.
    """
    overall_start = time.perf_counter()

    if not evaluation_report.results:
        raise SelectionError(
            "evaluation_report.results is empty; there are no candidates to select from."
        )

    if evaluation_report.failed_models:
        if mode is ExecutionMode.STRICT:
            raise SelectionError(
                f"STRICT mode: {len(evaluation_report.failed_models)} model(s) failed "
                f"upstream evaluation ({sorted(evaluation_report.failed_models)}); "
                f"refusing to select a champion from an incomplete field. "
                f"Use ExecutionMode.BEST_EFFORT to select among the remaining "
                f"successful candidates instead."
            )
        logger.warning(
            "BEST_EFFORT: excluding %d model(s) that failed evaluation: %s",
            len(evaluation_report.failed_models),
            sorted(evaluation_report.failed_models),
        )

    ranking = rank_candidates(evaluation_report.results)

    champion_candidate = ranking[0]
    champion_key = (
        champion_candidate.rmse,
        champion_candidate.mae,
        champion_candidate.r2,
    )
    tied_with = [c.name for c in ranking[1:] if (c.rmse, c.mae, c.r2) == champion_key]

    champion_result = SelectionResult(
        champion_name=champion_candidate.name,
        champion_evaluation=evaluation_report.results[champion_candidate.name],
        tied_with=tied_with,
    )

    total_duration = time.perf_counter() - overall_start

    report = SelectionReport(
        champion=champion_result,
        ranking=ranking,
        excluded_models=dict(evaluation_report.failed_models),
        mode=mode,
        total_duration_seconds=total_duration,
    )

    if tied_with:
        logger.info(
            "Champion selected: '%s' (tied on metrics with %s, won by insertion order), "
            "%d candidate(s) ranked, mode=%s, %.4fs",
            champion_result.champion_name,
            tied_with,
            len(ranking),
            mode.value,
            total_duration,
        )
    else:
        logger.info(
            "Champion selected: '%s', %d candidate(s) ranked, mode=%s, %.4fs",
            champion_result.champion_name,
            len(ranking),
            mode.value,
            total_duration,
        )

    return report
