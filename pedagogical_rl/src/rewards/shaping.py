"""
Reward composition -- paper formula (arXiv 2505.15607) + the shaping terms from
the reference implementation (eth-lre/PedagogicalRL classroom.py).

Base (paper):     r = r_sol + (r_ped - 1) * lambda
Shaping (code):   + end-of-conversation bonus  (+0.1 if tutor ended early via <end>)
                  + thinking reward            ((frac teacher turns with <think>) * 0.5)
                  - length penalty             (-0.5 if any turn hit the token limit)

  r_sol : post-dialog solve rate = fraction of K post-tutoring solutions passing ALL tests.
  r_ped : PRODUCT of binary judge verdicts -- ALL must accept (their code: any REJECT
          sets failed_judges=True). lambda: 0.75 standard / 1.0 thinking mode.

"hard" variant: any reject -> whole reward = -lambda (no shaping added).

Pure and model-free: takes already-computed ExecutionResults, judge verdicts, and a
few conversation-derived FEATURES (not the Conversation object), so it unit-tests
with mocks. classroom.py extracts those features and picks the judge-aggregation
policy. All shaping inputs default to off -> the clean paper formula.
"""

from __future__ import annotations

from dataclasses import dataclass

from .executor import ExecutionResult

# Defaults mirror the reference implementation.
DEFAULT_END_BONUS = 0.1
DEFAULT_THINKING_WEIGHT = 0.5
DEFAULT_LENGTH_PENALTY = 0.5


def solve_rate(results: list[ExecutionResult]) -> float:
    """Fraction of solutions that pass ALL their tests. Empty -> 0.0."""
    if not results:
        return 0.0
    return sum(r.all_passed for r in results) / len(results)


def thinking_fraction(teacher_messages: list[str]) -> float:
    """Fraction of teacher turns containing a proper <think>...</think> block."""
    if not teacher_messages:
        return 0.0
    used = sum(1 for m in teacher_messages if "<think>" in m and "</think>" in m)
    return used / len(teacher_messages)


@dataclass(frozen=True)
class RewardBreakdown:
    """The composed reward plus every ingredient, for logging/debugging."""

    total: float
    r_sol: float
    r_ped: float
    # shaping components (0.0 when their inputs are off)
    end_bonus: float
    thinking_reward: float
    length_penalty: float
    # diagnostics (not summed into `total`)
    pre_solve_rate: float
    post_solve_rate: float
    n_judges_accept: int
    n_judges_total: int

    @property
    def delta_solve_rate(self) -> float:
        """EVAL metric: how much tutoring improved the student."""
        return self.post_solve_rate - self.pre_solve_rate

    @property
    def pedagogy_passed(self) -> bool:
        return self.n_judges_total > 0 and self.n_judges_accept == self.n_judges_total


def compute_reward(
    pre_results: list[ExecutionResult],
    post_results: list[ExecutionResult],
    judge_verdicts: list[bool],
    *,
    lambda_ped: float = 0.75,
    hard: bool = False,
    # --- shaping features (extracted from the conversation by classroom.py) ---
    ended_early: bool = False,
    teacher_messages: list[str] | None = None,
    any_turn_truncated: bool = False,
    use_thinking: bool = False,
    # --- shaping weights ---
    end_bonus: float = DEFAULT_END_BONUS,
    thinking_weight: float = DEFAULT_THINKING_WEIGHT,
    length_penalty: float = DEFAULT_LENGTH_PENALTY,
) -> RewardBreakdown:
    """Compose the GRPO reward for one conversation.

    Args:
        pre_results: executor results for the UNTUTORED attempts (diagnostic only).
        post_results: executor results for the K post-dialog solutions -> r_sol.
        judge_verdicts: one bool per judge evaluation; True = accept.
        lambda_ped: pedagogy weight.
        hard: if True, any rejection collapses the reward to -lambda_ped.
        ended_early: True if the tutor emitted <end> before max_turns.
        teacher_messages: the tutor's turns, for measuring <think> usage.
        any_turn_truncated: True if any generation hit the per-turn token limit.
        use_thinking: apply the thinking reward (only in thinking mode).
        end_bonus / thinking_weight / length_penalty: shaping magnitudes.

    Returns:
        RewardBreakdown with `.total` as the scalar GRPO reward.
    """
    r_sol = solve_rate(post_results)

    n_total = len(judge_verdicts)
    n_accept = sum(judge_verdicts)
    all_accept = n_total > 0 and n_accept == n_total
    r_ped = 1.0 if all_accept else 0.0

    # Shaping terms (each 0.0 unless its trigger fired).
    end_term = end_bonus if ended_early else 0.0
    think_term = (
        thinking_weight * thinking_fraction(teacher_messages or [])
        if use_thinking else 0.0
    )
    length_term = -length_penalty if any_turn_truncated else 0.0

    if hard and not all_accept:
        # Hard pedagogy: a single rejection zeroes everything, shaping included.
        total = -lambda_ped
        end_term = think_term = length_term = 0.0
    else:
        base = r_sol + (r_ped - 1.0) * lambda_ped
        total = base + end_term + think_term + length_term

    return RewardBreakdown(
        total=total,
        r_sol=r_sol,
        r_ped=r_ped,
        end_bonus=end_term,
        thinking_reward=think_term,
        length_penalty=length_term,
        pre_solve_rate=solve_rate(pre_results),
        post_solve_rate=r_sol,
        n_judges_accept=n_accept,
        n_judges_total=n_total,
    )
