"""Replay benchmark contract; physical replay is never inferred from simulation."""

STATUS = "NOT_RUN"


def status() -> dict[str, str]:
    return {"status": STATUS, "simulated_replay": STATUS, "physical_replay": STATUS}

