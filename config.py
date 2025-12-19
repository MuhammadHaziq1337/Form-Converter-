
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific LLM model."""
    name: str
    context_tokens: int
    max_output_tokens: int
    tokens_per_char: float = 0.4 
    
    @property
    def chars_per_token(self) -> float:
        """Characters per token (inverse of tokens_per_char)."""
        return 1 / self.tokens_per_char


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for the extraction pipeline."""
    
    # Model settings
    model: ModelConfig
    
    # Token budgets
    system_prompt_tokens: int = 2_500
    safety_margin_tokens: int = 5_000
    
    
    safe_output_tokens: int = 64_000  
    min_sections_per_batch: int = 1
    max_sections_per_batch: int = 25  
    
    # Estimation constants
    estimated_tokens_per_field: int = 80
    estimated_tokens_per_table: int = 500
    estimated_tokens_per_section: int = 150
    
   
    rate_limit_retry_base_delay: float = 1.0  
    max_retries: int = 3
    
    # Concurrency
    concurrent_batch_limit: int = 5
    batch_group_delay: float = 1.0  
    
    # Chunking
    overlap_size: int = 4_000
    global_context_scan_tokens: int = 5_000
    
    # Debug
    debug_enabled: bool = False
    debug_dir: str = "debug_markdown"
    
    # Derived properties
    @property
    def available_input_tokens(self) -> int:

        return (
            self.model.context_tokens
            - self.system_prompt_tokens
            - self.safety_margin_tokens
        )
    
    @property
    def chunk_threshold_tokens(self) -> int:
        """Token count that triggers chunking."""
        return int(self.available_input_tokens * 0.90)
    
    @property
    def chunk_max_size_tokens(self) -> int:
        """Maximum chunk size in tokens."""
        return int(self.available_input_tokens * 0.95)
    
    @property
    def chunk_min_size_tokens(self) -> int:
        """Minimum chunk size in tokens."""
        return int(self.available_input_tokens * 0.50)
    
    @property
    def chunk_max_size_chars(self) -> int:
        """Maximum chunk size in characters."""
        return int(self.chunk_max_size_tokens * self.model.chars_per_token)
    
    @property
    def chunk_min_size_chars(self) -> int:
        """Minimum chunk size in characters."""
        return int(self.chunk_min_size_tokens * self.model.chars_per_token)
    
    @property
    def super_chunk_threshold_tokens(self) -> int:
        """Threshold for super-chunking (conservative default)."""
        return 200_000


# Predefined model configurations
GPT5_MINI = ModelConfig(
    name="gpt-5-mini",
    context_tokens=400_000,
    max_output_tokens=128_000,  
)

GPT5 = ModelConfig(
    name="gpt-5",
    context_tokens=1_000_000,  # 1M context window
    max_output_tokens=128_000,  # 128k output (API limit)
)

GPT4O = ModelConfig(
    name="gpt-4o-2024-08-06",
    context_tokens=128_000,
    max_output_tokens=4_096,
)

CLAUDE_SONNET = ModelConfig(
    name="claude-3-5-sonnet",
    context_tokens=200_000,
    max_output_tokens=8_192,
)


LARGE_DOC_THRESHOLD = 150_000 


def get_model_by_name(model_name: str) -> ModelConfig:
    """Get model config by name."""
    model_map = {
        "gpt-5-mini": GPT5_MINI,
        "gpt-5": GPT5,
        "gpt-4o-2024-08-06": GPT4O,
        "gpt-4o": GPT4O,
    }
    return model_map.get(model_name, GPT5)


def get_default_config() -> ExtractionConfig:
    """Get default configuration based on environment."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-5")
    debug = os.getenv("EXTRACTOR_DEBUG", "").lower() in ("1", "true", "yes")
    
    model = get_model_by_name(model_name)
    
    return ExtractionConfig(
        model=model,
        debug_enabled=debug,
    )


def get_config_for_document(doc_tokens: int, force_model: str = None) -> ExtractionConfig:

    debug = os.getenv("EXTRACTOR_DEBUG", "").lower() in ("1", "true", "yes")
    hybrid_mode = os.getenv("HYBRID_MODEL", "true").lower() in ("1", "true", "yes")
    
    if force_model:
        model = get_model_by_name(force_model)
    elif hybrid_mode and doc_tokens > LARGE_DOC_THRESHOLD:
        model = GPT5
        print(f"[Config] Hybrid Mode: Using GPT-5 for large document ({doc_tokens:,} tokens)")
    else:
        model = get_model_by_name(os.getenv("OPENAI_MODEL", "gpt-5"))
    
    # Adjust safe_output_tokens based on model
    safe_output = int(model.max_output_tokens * 0.5)
    
    return ExtractionConfig(
        model=model,
        debug_enabled=debug,
        safe_output_tokens=safe_output,
    )


# Global default (can be overridden)
DEFAULT_CONFIG = get_default_config()
