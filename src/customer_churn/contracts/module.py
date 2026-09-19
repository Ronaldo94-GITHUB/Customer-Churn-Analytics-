from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ModuleStatus(str, Enum):
    REGISTERED = "registered"
    PLANNED = "planned"
    DEVELOPMENT = "development"
    ACTIVE = "active"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ModuleMetadata:
    key: str
    name: str
    version: str
    description: str
    status: ModuleStatus


class PlatformModule(ABC):
    """Base contract for every platform module."""

    @property
    @abstractmethod
    def metadata(self) -> ModuleMetadata:
        """Return module metadata."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when the module is healthy."""
        raise NotImplementedError

