"""Codec robustness benchmark contract; experiments not run yet."""

STATUS = "NOT_RUN"
CODECS = ("mp3", "aac", "opus", "ogg")


def status() -> dict[str, object]:
    return {"status": STATUS, "codecs": list(CODECS), "reason": "codec/bitrate manifest not generated"}

