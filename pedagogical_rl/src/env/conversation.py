"""
Conversation: the state machine for ONE tutoring dialogue.

Holds the transcript + state for a single (tutor, student) dialogue about one
Problem. It does NOT call any models -- classroom.py drives it by asking whose
turn it is, generating that turn, and feeding the text back in. Keeping model
inference out of here makes the state logic pure and unit-testable.

Flow:
    START
      -> PRE_DIALOG_ATTEMPT   student solves ALONE (baseline solve rate)
      -> TEACHER_TURN <-> STUDENT_TURN   (dialogue, until <end> or max_turns)
      -> GENERATE_SOLUTION    student writes final code (K samples)
      -> DONE

Views: the tutor sees the reference solution (to guide) but is told never to
reveal it; the student never sees it. This asymmetry is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from ..data.mbpp import Problem

# The tutor emits this to end the dialogue early (chosen termination signal).
END_TOKEN = "<end>"


class Role(str, Enum):
    TEACHER = "teacher"
    STUDENT = "student"


class ConversationState(Enum):
    START = auto()
    PRE_DIALOG_ATTEMPT = auto()
    TEACHER_TURN = auto()
    STUDENT_TURN = auto()
    GENERATE_SOLUTION = auto()
    DONE = auto()


# --- Prompt templates -------------------------------------------------------
# Inline for now; move to configs/prompts/*.jinja when you start iterating on
# pedagogy wording. {tests} gives the student the function name/signature (MBPP
# convention); only the TEACHER template is given {solution}.
TEACHER_SYSTEM = """\
You are an expert programming tutor helping a student solve a Python problem.

Problem:
{prompt}

The student's code must eventually pass these tests:
{tests}

Reference solution (FOR YOUR EYES ONLY -- never reveal it, never paste working \
code, never write the full function for them):
{solution}

Teach by asking guiding questions, giving hints, and pointing out mistakes so the \
student writes the code THEMSELVES. When you believe the student can now solve it, \
end your message with {end_token}.
"""

STUDENT_SYSTEM = """\
You are a student learning to program in Python. Work with your tutor to solve this \
problem. Think aloud, ask questions, and try to write the solution yourself.

Problem:
{prompt}

Your solution must pass these tests:
{tests}
"""

# Appended as the final instruction when the student writes the final solution.
FINAL_SOLUTION_INSTRUCTION = (
    "Now write your final solution as a single Python function inside a "
    "```python code block."
)


@dataclass
class Conversation:
    """One tutoring dialogue and its state."""

    problem: Problem
    max_turns: int = 12  # hard backstop if the tutor never emits <end>

    state: ConversationState = ConversationState.START
    turn_count: int = 0
    # transcript entries: {"role": Role, "content": str}
    messages: list[dict] = field(default_factory=list)
    pre_dialog_solutions: list[str] = field(default_factory=list)
    post_dialog_solutions: list[str] = field(default_factory=list)

    # -- transitions ---------------------------------------------------------

    def record_pre_dialog(self, solutions: list[str]) -> None:
        """Store the untutored attempt(s), then open the dialogue with the tutor."""
        assert self.state == ConversationState.START, f"bad state {self.state}"
        self.pre_dialog_solutions = solutions
        self.state = ConversationState.TEACHER_TURN

    def add_teacher_message(self, text: str) -> None:
        assert self.state == ConversationState.TEACHER_TURN, f"bad state {self.state}"
        self.messages.append({"role": Role.TEACHER, "content": text})
        # End early on <end>, or fall through to the student's turn. The max_turns
        # backstop is checked after the student replies (see add_student_message).
        if END_TOKEN in text:
            self.state = ConversationState.GENERATE_SOLUTION
        else:
            self.state = ConversationState.STUDENT_TURN

    def add_student_message(self, text: str) -> None:
        assert self.state == ConversationState.STUDENT_TURN, f"bad state {self.state}"
        self.messages.append({"role": Role.STUDENT, "content": text})
        self.turn_count += 1
        # Hit the turn cap -> force the student to write the final solution.
        if self.turn_count >= self.max_turns:
            self.state = ConversationState.GENERATE_SOLUTION
        else:
            self.state = ConversationState.TEACHER_TURN

    def record_post_dialog(self, solutions: list[str]) -> None:
        """Store the K post-tutoring solutions and finish."""
        assert self.state == ConversationState.GENERATE_SOLUTION, f"bad state {self.state}"
        self.post_dialog_solutions = solutions
        self.state = ConversationState.DONE

    # -- queries -------------------------------------------------------------

    def next_speaker(self) -> Role | None:
        """Who generates next, or None if there's nothing to generate right now."""
        if self.state in (ConversationState.TEACHER_TURN,):
            return Role.TEACHER
        if self.state in (ConversationState.STUDENT_TURN, ConversationState.GENERATE_SOLUTION):
            return Role.STUDENT
        return None

    def is_over(self) -> bool:
        return self.state == ConversationState.DONE

    # -- rendering to chat format -------------------------------------------

    def render_for(self, role: Role) -> list[dict]:
        """Build chat messages (system + transcript) from `role`'s perspective.

        Each participant sees ITSELF as the assistant and the OTHER as the user,
        which is what the chat template expects.
        """
        if role == Role.TEACHER:
            system = TEACHER_SYSTEM.format(
                prompt=self.problem.prompt,
                tests="\n".join(self.problem.tests),
                solution=self.problem.reference_solution,
                end_token=END_TOKEN,
            )
        else:
            system = STUDENT_SYSTEM.format(
                prompt=self.problem.prompt,
                tests="\n".join(self.problem.tests),
            )

        chat = [{"role": "system", "content": system}]
        for m in self.messages:
            speaker = "assistant" if m["role"] == role else "user"
            chat.append({"role": speaker, "content": m["content"]})

        # When the student is about to produce the final solution, nudge it explicitly.
        if role == Role.STUDENT and self.state == ConversationState.GENERATE_SOLUTION:
            chat.append({"role": "user", "content": FINAL_SOLUTION_INSTRUCTION})

        return chat

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "task_id": self.problem.task_id,
            "state": self.state.name,
            "turn_count": self.turn_count,
            "messages": [{"role": m["role"].value, "content": m["content"]} for m in self.messages],
            "pre_dialog_solutions": self.pre_dialog_solutions,
            "post_dialog_solutions": self.post_dialog_solutions,
        }
