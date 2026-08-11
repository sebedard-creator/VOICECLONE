class VoiceCloneError(Exception):
    """Base exception shown as a clear operational error in the UI."""


class RootPathUnavailable(VoiceCloneError):
    """Raised when the configured runtime disk/root path is unavailable."""


class DependencyMissing(VoiceCloneError):
    """Raised when an external runtime dependency is missing."""


class ConfigurationError(VoiceCloneError):
    """Raised when a required configuration value is missing or invalid."""

