# src/prompts.py
# Shared prompt-building logic used by all training notebooks.

SYSTEM_PROMPT = (
    "You are an AI tutoring assistant for an Intelligent Tutoring System. "
    "When given a [NEGATIVE] prompt, generate Python code that contains bugs "
    "appropriate to the specified student persona or bug type. "
    "When given a [POSITIVE] prompt, generate correct, clean Python code."
)

PERSONALITY_DESCRIPTIONS = {
    "CONFUSED_STUDENT": (
        "a confused student who misunderstands the problem requirements "
        "and writes code that doesn't quite match what's being asked"
    ),
    "SYNTAX_STRUGGLER": (
        "a student who struggles with Python syntax and makes errors like "
        "wrong indentation, missing colons, or incorrect operator usage"
    ),
    "IMPATIENT_STUDENT": (
        "an impatient student who writes code quickly without thinking it "
        "through, often missing edge cases or making off-by-one errors"
    ),
    "PROGRAMMING_HELPER": (
        "a well-meaning but inexperienced helper who writes mostly-correct "
        "code but introduces subtle logical bugs"
    ),
    "OVERCONFIDENT_WRONG": (
        "an overconfident student who writes code that looks polished and "
        "complete but contains fundamental logical errors"
    ),
}


def build_personality_negative_prompt(problem: str, personality: str) -> str:
    """NEGATIVE prompt conditioned on student personality (strategy1)."""
    desc = PERSONALITY_DESCRIPTIONS.get(
        personality,
        f"a {personality.lower().replace('_', ' ')} student"
    )
    return (
        f"[NEGATIVE] Generate a Python function for the following problem "
        f"as it would be written by {desc}.\n\n"
        f"Problem: {problem.strip()}"
    )


def build_positive_prompt(problem: str) -> str:
    return (
        f"[POSITIVE] Generate a correct Python function for the following problem.\n\n"
        f"Problem: {problem.strip()}"
    )


def make_example(user_prompt: str, code: str) -> dict:
    return {
        "conversations": [
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": code},
        ]
    }
