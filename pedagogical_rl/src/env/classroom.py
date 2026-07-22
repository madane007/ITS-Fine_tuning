"""
Classroom: batched rollout orchestrator -- ties every component together.

Given problems, it runs a GRPO GROUP of conversations per problem (group_size G),
advancing ALL conversations across ALL problems together in batches for GPU
efficiency, then scores each into a reward. Returns results grouped by problem so
grpo.py can compute group-relative advantages.

Two counts to keep distinct:
  G  (group_size)  = conversations per problem  -> the GRPO comparison group
  K  (k_solutions) = student solutions sampled at the END of each conversation
                     for the post-dialog solve rate r_sol (paper: K=8)

Pipeline per conversation:
  pre-dialog (untutored, per-problem) -> tutor<->student dialogue -> K final
  solutions -> executor scores them (r_sol) -> judges rate the transcript (r_ped)
  -> shaping.compute_reward -> RolloutResult.

Only calls engines' .generate/.k_samples, so swapping transformers->vLLM changes
nothing here.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..data.mbpp import Problem
from ..rewards.executor import ExecutionResult
from ..rewards.functional import functional_reward
from ..rewards.shaping import RewardBreakdown, compute_reward, solve_rate
from .conversation import (
    Conversation,
    ConversationState,
    Role,
    END_TOKEN,
    STUDENT_SYSTEM,
    FINAL_SOLUTION_INSTRUCTION,
)

# --- Judge rubrics (the paper's two: Answer Leakage + Helpfulness) ----------
# Each judge sees the problem, the gold solution, and the transcript, and must
# end with a verdict line. ACCEPT = good pedagogy; any REJECT zeroes r_ped.
_JUDGE_LEAKAGE = """\
You are grading a programming tutor for ANSWER LEAKAGE.

Problem:
{prompt}

Reference solution (what the tutor must NOT hand over):
{solution}

Tutoring transcript:
{transcript}

Did the tutor give away the solution -- pasting working code or dictating the full
answer instead of guiding the student to write it? Reply REJECT if the tutor leaked
the solution, ACCEPT if it guided without leaking.
End with exactly one line: "VERDICT: ACCEPT" or "VERDICT: REJECT".
"""

_JUDGE_HELPFULNESS = """\
You are grading a programming tutor for HELPFULNESS.

Problem:
{prompt}

Tutoring transcript:
{transcript}

Was the tutoring constructive, relevant, and encouraging -- moving the student
toward the solution with useful hints and questions? Reply ACCEPT if helpful,
REJECT if unhelpful, confusing, or off-topic.
End with exactly one line: "VERDICT: ACCEPT" or "VERDICT: REJECT".
"""

_JUDGE_RUBRICS = (_JUDGE_LEAKAGE, _JUDGE_HELPFULNESS)

# Sandbox scoring (executor.py) spawns one subprocess per candidate solution and
# blocks on it -- fully CPU-bound, GPU idle the whole time. subprocess.run releases
# the GIL while waiting, so a thread pool overlaps these instead of running them one
# at a time (no need for a process pool).
_EXECUTOR_WORKERS = 8


@dataclass
class RolloutResult:
    """One conversation's outcome."""

    problem: Problem
    conversation: Conversation
    reward: RewardBreakdown


class Classroom:
    def __init__(
        self,
        tutor,
        student,
        judge,
        *,
        group_size: int = 8,       # G: conversations per problem (GRPO group)
        k_solutions: int = 8,      # K: student solutions for the solve rate
        judge_samples: int = 2,    # samples per rubric (paper: 2 -> 2 rubrics x 2 = 4)
        max_turns: int = 12,
        lambda_ped: float = 0.75,
        hard: bool = False,
        use_thinking: bool = False,
    ):
        self.tutor = tutor
        self.student = student
        self.judge = judge
        self.group_size = group_size
        self.k_solutions = k_solutions
        self.judge_samples = judge_samples
        self.max_turns = max_turns
        self.lambda_ped = lambda_ped
        self.hard = hard
        self.use_thinking = use_thinking

    # -- public API ----------------------------------------------------------

    def rollout_batch(self, problems: list[Problem]) -> list[list[RolloutResult]]:
        """Run G conversations per problem, batched. Returns [problem][group]."""
        # 1. Pre-dialog solve rate is a per-PROBLEM property (untutored, no tutor
        #    dependence), so compute it once per problem, batched, and share it.
        pre_results = self._pre_dialog(problems)

        # 2. Build G conversations per problem; open each with the shared pre-dialog.
        convs: list[Conversation] = []
        owner: list[int] = []  # convs[i] belongs to problems[owner[i]]
        for pi, problem in enumerate(problems):
            for _ in range(self.group_size):
                c = Conversation(problem, max_turns=self.max_turns)
                c.record_pre_dialog([])  # just advances state; pre score is per-problem
                convs.append(c)
                owner.append(pi)

        # 3. Advance ALL conversations together until every one is DONE.
        self._run_dialogues(convs)

        # 4. Score post-dialog code + judge each transcript (both batched).
        post_results = self._score_solutions(convs)
        verdicts = self._judge_all(convs)

        # 5. Compose rewards and regroup by problem.
        grouped: list[list[RolloutResult]] = [[] for _ in problems]
        for i, c in enumerate(convs):
            pi = owner[i]
            teacher_msgs = [m["content"] for m in c.messages if m["role"] == Role.TEACHER]
            reward = compute_reward(
                pre_results[pi],
                post_results[i],
                verdicts[i],
                lambda_ped=self.lambda_ped,
                hard=self.hard,
                ended_early=any(END_TOKEN in m for m in teacher_msgs),
                teacher_messages=teacher_msgs,
                use_thinking=self.use_thinking,
            )
            grouped[pi].append(RolloutResult(c.problem, c, reward))
        return grouped

    # -- stages --------------------------------------------------------------

    def _pre_dialog(self, problems: list[Problem]) -> list[list[ExecutionResult]]:
        """Untutored student attempt per problem -> executor results (batched)."""
        chats = [self._solo_chat(p) for p in problems]
        sampled = self.student.generate(chats, n=self.k_solutions)  # [prob][K]
        return self._score_grouped(zip(problems, sampled))

    def _run_dialogues(self, convs: list[Conversation]) -> None:
        """Batched turn-taking: each step, bucket conversations by whose turn it
        is, generate that bucket in one batched call, feed results back."""
        while not all(c.is_over() for c in convs):
            teacher = [c for c in convs if c.state == ConversationState.TEACHER_TURN]
            student = [c for c in convs if c.state == ConversationState.STUDENT_TURN]
            final = [c for c in convs if c.state == ConversationState.GENERATE_SOLUTION]

            if teacher:
                outs = self.tutor.generate([c.render_for(Role.TEACHER) for c in teacher], n=1)
                for c, o in zip(teacher, outs):
                    c.add_teacher_message(o[0])
            if student:
                outs = self.student.generate([c.render_for(Role.STUDENT) for c in student], n=1)
                for c, o in zip(student, outs):
                    c.add_student_message(o[0])
            if final:
                outs = self.student.generate(
                    [c.render_for(Role.STUDENT) for c in final], n=self.k_solutions
                )
                for c, o in zip(final, outs):
                    c.record_post_dialog(o)

    def _score_solutions(self, convs: list[Conversation]) -> list[list[ExecutionResult]]:
        """Run each conversation's K post-dialog solutions through the executor."""
        return self._score_grouped((c.problem, c.post_dialog_solutions) for c in convs)

    @staticmethod
    def _score_grouped(
        groups: Iterable[tuple[Problem, list[str]]],
    ) -> list[list[ExecutionResult]]:
        """Score every (problem, solution) pair across all groups concurrently, then
        regroup -- see _EXECUTOR_WORKERS for why this is threaded."""
        groups = list(groups)
        flat = [(p, s) for p, sols in groups for s in sols]
        with ThreadPoolExecutor(max_workers=_EXECUTOR_WORKERS) as pool:
            flat_results = list(pool.map(lambda ps: functional_reward(*ps), flat))
        out, i = [], 0
        for _, sols in groups:
            out.append(flat_results[i : i + len(sols)])
            i += len(sols)
        return out

    def _judge_all(self, convs: list[Conversation]) -> list[list[bool]]:
        """Return per-conversation verdict lists (rubrics x judge_samples), batched
        one rubric at a time."""
        verdicts: list[list[bool]] = [[] for _ in convs]
        for rubric in _JUDGE_RUBRICS:
            chats = [self._judge_chat(rubric, c) for c in convs]
            outs = self.judge.generate(chats, n=self.judge_samples, greedy=False)
            for i, samples in enumerate(outs):
                verdicts[i].extend(_parse_verdict(s) for s in samples)
        return verdicts

    # -- prompt builders -----------------------------------------------------

    def _solo_chat(self, problem: Problem) -> list[dict]:
        """Untutored student prompt: solve the problem with no tutor."""
        system = STUDENT_SYSTEM.format(
            prompt=problem.prompt, tests="\n".join(problem.tests)
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": FINAL_SOLUTION_INSTRUCTION},
        ]

    def _judge_chat(self, rubric: str, conv: Conversation) -> list[dict]:
        content = rubric.format(
            prompt=conv.problem.prompt,
            solution=conv.problem.reference_solution,
            transcript=_transcript_text(conv),
        )
        return [{"role": "user", "content": content}]


# --- helpers ----------------------------------------------------------------

def _transcript_text(conv: Conversation) -> str:
    lines = []
    for m in conv.messages:
        who = "TUTOR" if m["role"] == Role.TEACHER else "STUDENT"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


def _parse_verdict(text: str) -> bool:
    """True = ACCEPT (good pedagogy). Default to REJECT on unparseable output so we
    never reward pedagogy we couldn't verify."""
    upper = text.upper()
    idx = upper.rfind("VERDICT:")
    tail = upper[idx:] if idx != -1 else upper
    if "REJECT" in tail:
        return False
    if "ACCEPT" in tail:
        return True
    return False
