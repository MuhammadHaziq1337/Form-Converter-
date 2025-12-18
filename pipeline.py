"""Extraction pipeline orchestration."""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import time
import asyncio

from schemas import FormDocument, FormSection, DocumentStructure, DocumentMetadata
from config import ExtractionConfig, DEFAULT_CONFIG


@dataclass
class PipelineMetrics:
    """Metrics collected during extraction."""
    input_tokens: int = 0
    output_tokens: int = 0
    batches_processed: int = 0
    sections_extracted: int = 0
    fields_extracted: int = 0
    duration_seconds: float = 0.0
    strategy_used: str = ""


@dataclass
class PipelineResult:
    """Result of the extraction pipeline."""
    document: FormDocument
    metrics: PipelineMetrics
    validation: dict
    is_complete: bool
    warnings: List[str] = field(default_factory=list)


class ExtractionPipeline:

    def __init__(
        self, 
        markdown: str, 
        config: ExtractionConfig = DEFAULT_CONFIG
    ):
        self.markdown = markdown
        self.config = config
        self.metrics = PipelineMetrics()
        self.warnings: List[str] = []
        
        # Intermediate state
        self._strategy = None
        self._structure: Optional[DocumentStructure] = None
        self._sections: List[FormSection] = []
        
    def run(self) -> PipelineResult:
        """Execute the full extraction pipeline."""
        start_time = time.time()
        
        try:
            # Import here to avoid circular dependency
            from extractor import (
                estimate_tokens, estimate_output_tokens, DocumentStrategy,
                _extract_full_document, _clean_markdown,
                extract_structure, validate_structure_coverage,
                group_sections_into_batches,
                merge_batch_results, validate_extraction_completeness,
                _extract_metadata_regex
            )
            
            # Step 0: Clean markdown (remove noise like page numbers, repeated headers)
            original_len = len(self.markdown)
            #self.markdown = _clean_markdown(self.markdown)
            cleaned_len = len(self.markdown)
            if original_len != cleaned_len:
                self._log(f"Cleaned markdown: {original_len:,} → {cleaned_len:,} chars ({original_len - cleaned_len:,} removed)")
            
            # Step 1: Analyze and save debug info
            self._analyze_document(estimate_tokens, estimate_output_tokens)
            self._save_debug_input()
            
            # Step 1: Determine strategy
            self._determine_strategy(DocumentStrategy)
            
            # Step 2: Execute appropriate strategy
            if self._strategy.mode == "single_pass":
                try:
                    document = self._execute_single_pass(_extract_full_document)
                except ValueError as e:
                    # Single-pass hit output limit (usually due to reasoning tokens)
                    # Fall back to targeted extraction
                    if "length limit was reached" in str(e):
                        self._log("  Single-pass truncated, retrying with targeted extraction...")
                        self.metrics.strategy_used = "targeted_extraction_fallback"
                        document = self._execute_targeted_extraction(
                            extract_structure,
                            validate_structure_coverage,
                            group_sections_into_batches,
                            merge_batch_results,
                            _extract_metadata_regex
                        )
                    else:
                        raise
            else:
                document = self._execute_targeted_extraction(
                    extract_structure,
                    validate_structure_coverage,
                    group_sections_into_batches,
                    merge_batch_results,
                    _extract_metadata_regex
                )
            
            # Step 3: Validate
            validation = self._validate(document, validate_extraction_completeness)
            
            # Finalize metrics
            self.metrics.duration_seconds = time.time() - start_time
            self.metrics.sections_extracted = len(document.sections)
            
            return PipelineResult(
                document=document,
                metrics=self.metrics,
                validation=validation,
                is_complete=validation.get("is_complete", False),
                warnings=self.warnings,
            )
            
        except Exception as e:
            self.metrics.duration_seconds = time.time() - start_time
            raise ExtractionError(f"Pipeline failed: {e}", metrics=self.metrics) from e
    
    def _analyze_document(self, estimate_tokens, estimate_output_tokens) -> None:
        """Analyze document and populate initial metrics."""
        self.metrics.input_tokens = estimate_tokens(self.markdown)
        output_estimate = estimate_output_tokens(self.markdown)
        self._log(f"Document: {len(self.markdown):,} chars ({self.metrics.input_tokens:,} tokens)")
        self._log(f"Estimated output: {output_estimate:,} tokens")
    
    def _save_debug_input(self) -> None:
        """Save input markdown for debugging if enabled."""
        if not self.config.debug_enabled:
            return
            
        debug_dir = Path(self.config.debug_dir)
        debug_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = debug_dir / f"full_markdown_{timestamp}.md"
        path.write_text(self.markdown, encoding="utf-8")
        self._log(f"💾 Saved debug markdown to: {path}")
    
    def _determine_strategy(self, DocumentStrategy) -> None:
        """Determine optimal extraction strategy."""
        self._strategy = DocumentStrategy.determine(
            doc_tokens=self.metrics.input_tokens,
            model_context=self.config.model.context_tokens,
            max_output=self.config.model.max_output_tokens,
        )
        
        self.metrics.strategy_used = self._strategy.mode
        self._log(f"Strategy: {self._strategy.mode} (utilization: {self._strategy.utilization_ratio:.1%})")
    
    def _execute_single_pass(self, _extract_full_document) -> FormDocument:
        """Execute single-pass extraction for small documents."""
        self._log("Mode: Single Pass (best accuracy)")
        self._log("═══════════════════════════════════════════")
        return _extract_full_document(self.markdown)
    
    def _execute_targeted_extraction(
        self,
        extract_structure,
        validate_structure_coverage,
        group_sections_into_batches,
        merge_batch_results,
        _extract_metadata_regex
    ) -> FormDocument:
        """Execute targeted extraction for large documents using continuous streaming."""
        self._log("Mode: Continuous Streaming (Zero-Gap)")
        self._log("═══════════════════════════════════════════")
        
        # Use the new streaming extraction
        from extractor import execute_streaming_extraction
        
        self._sections = execute_streaming_extraction(self.markdown, self.config)
        
        self.metrics.batches_processed = 1  # Streaming is one logical operation
        
        # Build final document
        return self._build_document(_extract_metadata_regex)
    
    def _build_document(self, _extract_metadata_regex) -> FormDocument:
        """Build final FormDocument from extracted sections."""
        regex_metadata = _extract_metadata_regex(self.markdown)
        metadata = DocumentMetadata(**regex_metadata) if regex_metadata else DocumentMetadata()
        
        # Derive document title from structure, metadata, or first section
        if self._structure and self._structure.document_title:
            doc_title = self._structure.document_title
        elif metadata.document_name:
            doc_title = metadata.document_name
        elif self._sections:
            doc_title = self._sections[0].title
        else:
            doc_title = "Untitled Document"
        
        return FormDocument(
            metadata=metadata,
            name=doc_title,
            description=f"Extracted form with {len(self._sections)} sections",
            filled_by="mixed",
            sections=self._sections,
        )
    
    def _validate(self, document: FormDocument, validate_extraction_completeness) -> dict:
        """Validate extraction completeness."""
        self._log("Step 5: Validating completeness...")
        validation = validate_extraction_completeness(self.markdown, document.sections)
        self.metrics.fields_extracted = validation["extracted_count"]
        
        if validation["is_complete"]:
            self._log(f"  ✓ Coverage: {validation['coverage_percentage']:.1f}%")
        else:
            self._log(f"    Coverage: {validation['coverage_percentage']:.1f}%")
            self.warnings.extend(validation["warnings"])
            for warning in validation["warnings"]:
                self._log(f"    {warning}")
        
        return validation
    
    def _log_structure(self) -> None:
        """Log extracted structure details."""
        if not self._structure:
            return
        self._log(f"  Found {self._structure.total_sections} sections:")
        for s in self._structure.sections:
            self._log(f"    • {s.title} ({s.estimated_density}, ~{s.estimated_field_count} fields)")
    
    def _log(self, message: str) -> None:
        """Log a message."""
        print(f"[Extractor] {message}")


class ExtractionError(Exception):
    """Raised when extraction fails."""
    def __init__(self, message: str, metrics: Optional[PipelineMetrics] = None):
        super().__init__(message)
        self.metrics = metrics
