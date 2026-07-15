"""
Sandboxed Python execution harness -- the correctness oracle for the RL loop.

It runs model-generated code against MBPP-style asserts in an isolated subprocess
and reports how many passed. Because the code is hostile (an RL policy will try to
reward-hack), it must run out-of-process, time-limited, and (on Linux) resource-capped.

This layer gives PROCESS + TIMEOUT isolation (and CPU/memory limits on POSIX). It does
NOT isolate the network or full filesystem -- for untrusted code at scale, run this whole
harness inside a container/nsjail. On Windows only the timeout applies: develop here, but
run real training on Linux/WSL2.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# The child prints its JSON result behind this marker, so we can find our result
# even if the student's own code writes noise to stdout.
_RESULT_MARKER = "__PEDCODER_RESULT__"


@dataclass(frozen=True)
class ExecutionResult:
    passed: int
    total: int
    timed_out: bool = False
    crashed: bool = False
    error: str | None = None

    @property
    def reward(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total


# --- The harness that runs INSIDE the child process -------------------------
# It reads {code, setup_code, tests} as JSON from stdin, applies resource limits
# (POSIX only), runs each assert counting passes, and prints one JSON line behind
# the marker. Stdlib-only so it imports cleanly under the same interpreter.
_HARNESS = '''\
import json, sys

def _apply_limits(mem_bytes):
    try:
        import resource
    except ImportError:
        return                      # Windows: rely on the parent's wall-clock timeout
    if mem_bytes:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

cfg = json.loads(sys.stdin.read())
_apply_limits(cfg["memory_bytes"])

ns = {{"__name__": "__candidate__"}}

# setup + student code share one namespace, like MBPP expects the target function
# to be defined at module scope. A crash here means the code didn't even define/run.
try:
    exec(cfg["setup_code"], ns)
    exec(cfg["code"], ns)
except Exception as e:
    print("{marker}" + json.dumps(
        {{"passed": 0, "total": len(cfg["tests"]),
          "error": f"{{type(e).__name__}}: {{e}}", "crashed": True}}))
    sys.exit(0)

passed = 0
first_error = None
for t in cfg["tests"]:
    try:
        exec(t, ns)
        passed += 1
    except Exception as e:
        if first_error is None:
            first_error = f"{{type(e).__name__}}: {{e}}"

print("{marker}" + json.dumps(
    {{"passed": passed, "total": len(cfg["tests"]),
      "error": first_error, "crashed": False}}))
'''


def run_tests(
    code: str,
    tests: list[str],
    *,
    setup_code: str = "",
    timeout_s: float = 10.0,
    memory_mb: int = 512,
) -> ExecutionResult:
    """Run `code` against `tests` in an isolated subprocess and score it.

    Args:
        code: Candidate solution (usually a single function definition).
        tests: Assert statements that raise on failure (MBPP `test_list`).
        setup_code: Imports/fixtures run before the candidate (MBPP `test_setup_code`).
        timeout_s: Wall-clock kill deadline for the whole run.
        memory_mb: Address-space cap enforced on POSIX; ignored on Windows.

    Returns:
        ExecutionResult -- never raises; every failure mode maps to a result.
    """
    if not tests:
        return ExecutionResult(passed=0, total=0, error="no tests provided")

    # 1. Pack the inputs into one JSON string for the child's stdin.
    payload = json.dumps({
        "code": code,
        "setup_code": setup_code,
        "tests": tests,
        "memory_bytes": memory_mb * 1024 * 1024 if memory_mb else 0,
    })

    # A stripped env: no inherited secrets, no bytecode files, unbuffered IO.
    child_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    # 2. Write the harness into a throwaway temp dir (also the child's cwd, so any
    #    files the student writes land here and get deleted with the dir).
    with tempfile.TemporaryDirectory(prefix="pedcoder_") as workdir:
        harness_path = Path(workdir) / "_harness.py"
        harness_path.write_text(_HARNESS.format(marker=_RESULT_MARKER), encoding="utf-8")

        # 3. Spawn the child and wait, with the timeout as the kill switch.
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(harness_path)],  # -I isolates the interpreter
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=workdir,
                env=child_env,
                start_new_session=(os.name == "posix"),  # own process group for clean kill
            )
        except subprocess.TimeoutExpired:
            # A timeout is a normal outcome (infinite loop / too slow), not an error
            # in US -- convert it to a zero-reward result instead of propagating.
            return ExecutionResult(
                passed=0, total=len(tests), timed_out=True,
                error=f"timeout after {timeout_s}s",
            )

    # 4. Find our marked line in stdout and parse it back into a result.
    return _parse_output(completed.stdout, completed.stderr, len(tests))


def _parse_output(stdout: str, stderr: str, total: int) -> ExecutionResult:
    """Extract the sentinel JSON line; missing output means the child died hard."""
    for line in stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            data = json.loads(line[len(_RESULT_MARKER):])
            return ExecutionResult(
                passed=data["passed"],
                total=data["total"],
                error=data.get("error"),
                crashed=data.get("crashed", False),
            )

    # No marker -> segfault, OOM kill, or rlimit killed it before it could report.
    detail = stderr.strip().splitlines()[-1] if stderr.strip() else "no output"
    return ExecutionResult(
        passed=0, total=total, crashed=True,
        error=f"candidate crashed: {detail}",
    )
