from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, List

T = TypeVar('T')

@dataclass
class ExtractionResult(Generic[T]):

    value: Optional[T] = None
    success: bool = True
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def ok(cls, value: T, warnings: Optional[List[str]] = None) -> 'ExtractionResult[T]':
        """Create a successful result."""
        return cls(value=value, success=True, warnings=warnings or [])
    
    @classmethod
    def fail(cls, error: str, warnings: Optional[List[str]] = None) -> 'ExtractionResult[T]':
        """Create a failed result."""
        return cls(value=None, success=False, error=error, warnings=warnings or [])
    
    @classmethod
    def partial(cls, value: T, error: str, warnings: Optional[List[str]] = None) -> 'ExtractionResult[T]':
        """Create a partial success (got something but not complete)."""
        return cls(value=value, success=False, error=error, warnings=warnings or [])
    
    def unwrap(self) -> T:
        """Get value or raise if failed."""
        if not self.success or self.value is None:
            raise ValueError(self.error or "Result has no value")
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Get value or return default if failed."""
        if self.value is None:
            return default
        return self.value
    
    def map(self, func) -> 'ExtractionResult':
        """Transform the value if successful."""
        if self.success and self.value is not None:
            try:
                new_value = func(self.value)
                return ExtractionResult.ok(new_value, self.warnings)
            except Exception as e:
                return ExtractionResult.fail(str(e), self.warnings)
        return self
    
    def add_warning(self, warning: str) -> 'ExtractionResult[T]':
        """Add a warning to the result."""
        self.warnings.append(warning)
        return self
