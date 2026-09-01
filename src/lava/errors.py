"""User-facing LAVA error types and actionable dependency messages."""


class LAVAError(RuntimeError):
    """Base error for an actionable LAVA contract violation."""


class DetectorNotFoundError(LAVAError):
    pass


class ArtifactNotReadyError(LAVAError):
    pass


class FrameworkDependencyError(LAVAError):
    pass

