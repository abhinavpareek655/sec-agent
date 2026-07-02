"""
Wraps orchestrator.run with objective, tool-verified success grading.

The orchestrator's own "success" flag only reflects whether the agent
completed its steps without erroring — it says nothing about whether any
challenge actually got solved. This wrapper polls the real scoreboard state
before and after the run and grades on that, ignoring the agent's own
self-report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class GradedResult:
    real_success: bool
    solved_before: int
    solved_after: int
    newly_solved: list[dict]
    total_challenges: int
    agent_result: object
    agent_reported_success: bool | None = None
    error: str | None = None
    raw_baseline: dict = field(default_factory=dict, repr=False)
    raw_final: dict = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        if self.error:
            return f"GRADING ERROR: {self.error}"

        verdict = "SOLVED" if self.real_success else "NO PROGRESS"
        lines = [
            f"[{verdict}] solved_count: {self.solved_before} -> {self.solved_after} "
            f"(+{self.solved_after - self.solved_before}) out of {self.total_challenges}",
        ]
        if self.newly_solved:
            names = ", ".join(c.get("name", str(c.get("id"))) for c in self.newly_solved)
            lines.append(f"Newly solved: {names}")
        if self.agent_reported_success is not None and self.agent_reported_success != self.real_success:
            lines.append(
                f"NOTE: agent self-reported success={self.agent_reported_success}, "
                f"which disagrees with the actual scoreboard result."
            )
        return "\n".join(lines)


def run_graded(
    orchestrator,
    scoreboard_tool,
    scoreboard_url: str,
    goal: str,
    max_steps: int = 30,
    token: str | None = None,
) -> GradedResult:
    """
    Runs orchestrator.run(goal=goal, max_steps=max_steps) but grades success
    by directly polling scoreboard_tool before and after, rather than trusting
    whatever the orchestrator/agent reports about itself.

    scoreboard_tool must be a ScoreboardPollTool instance (or expose the same
    .execute(url=..., token=..., reset=...) -> dict interface).
    """

    # Baseline poll, forced fresh so it isn't polluted by a snapshot the
    # agent itself may have already taken earlier in the process lifetime.
    baseline = scoreboard_tool.execute(url=scoreboard_url, token=token, reset=True)
    if not baseline.get("success"):
        return GradedResult(
            real_success=False,
            solved_before=-1,
            solved_after=-1,
            newly_solved=[],
            total_challenges=-1,
            agent_result=None,
            error=f"Baseline poll failed: {baseline.get('error')}",
        )

    solved_before = baseline["solved_count"]
    total_challenges = baseline["challenge_count"]

    agent_result = None
    agent_reported_success = None
    run_error = None
    try:
        agent_result = orchestrator.run(goal=goal, max_steps=max_steps)
        if isinstance(agent_result, dict):
            agent_reported_success = agent_result.get("success")
    except Exception as exc:
        run_error = str(exc)

    # Final poll against the same snapshot key (reset=False) so we get a
    # real diff, not just two independent baselines.
    final = scoreboard_tool.execute(url=scoreboard_url, token=token, reset=False)
    if not final.get("success"):
        return GradedResult(
            real_success=False,
            solved_before=solved_before,
            solved_after=-1,
            newly_solved=[],
            total_challenges=total_challenges,
            agent_result=agent_result,
            agent_reported_success=agent_reported_success,
            error=run_error or f"Final poll failed: {final.get('error')}",
            raw_baseline=baseline,
        )

    solved_after = final["solved_count"]
    newly_solved = final.get("newly_solved", [])

    return GradedResult(
        real_success=(solved_after > solved_before) or bool(newly_solved),
        solved_before=solved_before,
        solved_after=solved_after,
        newly_solved=newly_solved,
        total_challenges=total_challenges,
        agent_result=agent_result,
        agent_reported_success=agent_reported_success,
        error=run_error,
        raw_baseline=baseline,
        raw_final=final,
    )