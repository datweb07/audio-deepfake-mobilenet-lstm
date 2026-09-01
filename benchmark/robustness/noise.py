"""Controlled additive-noise benchmark contract; experiments not run yet."""

STATUS = "NOT_RUN"
SNR_DB_CONDITIONS = (20, 10, 5, 0)


def status() -> dict[str, object]:
    return {"status": STATUS, "conditions": list(SNR_DB_CONDITIONS), "reason": "condition manifest not generated"}

