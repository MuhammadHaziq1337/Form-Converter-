from typing import List, Optional, Literal, Union, Dict
from pydantic import BaseModel, Field


# 0. GLOBAL CONTEXT (Document-wide definitions, rules, instructions)
class GlobalContext(BaseModel):
    """Critical document-wide context that must be preserved across all chunks."""
    definitions: Optional[str] = Field(None, description="Key term definitions (e.g., 'The Applicant refers to...')")
    instructions: Optional[str] = Field(None, description="Global instructions and rules (e.g., 'If you answer No to Q1, skip Section 5')")
    completion_rules: Optional[str] = Field(None, description="Document completion logic (e.g., 'Sections marked * are mandatory')")
    special_notes: Optional[str] = Field(None, description="Important notes or warnings that apply to the entire document")


# 1. METADATA
class DocumentMetadata(BaseModel):
    document_name: Optional[str] = Field(None, description="Document title from header/footer")
    version: Optional[str] = Field(None, description="Version number if present")
    organization: Optional[str] = Field(None, description="Full organization name")
    cricos_code: Optional[str] = Field(None, description="CRICOS Code if present")
    provider_code: Optional[str] = Field(None, description="Provider Code if present")
    abn: Optional[str] = Field(None, description="ABN if present")


# 2. TABLE STRUCTURE (Explicit column mapping with positional fallback)
class TableCell(BaseModel):
    """A single cell with explicit column reference for data integrity."""
    column: str = Field(..., description="Column header this cell belongs to (prevents positional errors)")
    value: Optional[str] = Field(None, description="Cell value, null if empty")
    input_type: Optional[Literal["text", "checkbox", "date", "select", "radio"]] = Field(None, description="Interactive cell type if editable")
    options: Optional[List[str]] = Field(None, description="Options for radio/select cells (e.g., ['Yes', 'No'])")
    editable: bool = Field(default=False, description="True if cell needs user input")


class TableRow(BaseModel):
    """A row with explicitly labeled cells - each cell knows its column."""
    cells: List[TableCell] = Field(..., description="Cells with explicit column references")


class FormTable(BaseModel):
    """Table with explicit column-to-value mapping."""
    id: Optional[str] = Field(None, description="Section-scoped ID (auto-generated if null)")
    type: Literal["table"] = "table"
    title: Optional[str] = Field(None, description="Table caption/title if present")
    headers: List[str] = Field(..., description="Column headers in order")
    rows: List[TableRow] = Field(..., description="Rows with explicit column references")


# 3. VALIDATION RULES
class ValidationRule(BaseModel):
    """Validation constraints for a field."""
    min_length: Optional[int] = Field(None, description="Minimum text length")
    max_length: Optional[int] = Field(None, description="Maximum text length")
    pattern: Optional[str] = Field(None, description="Regex pattern for text fields")
    min_value: Optional[float] = Field(None, description="Minimum value for number fields")
    max_value: Optional[float] = Field(None, description="Maximum value for number fields")
    min_date: Optional[str] = Field(None, description="Minimum date (ISO format or 'today')")
    max_date: Optional[str] = Field(None, description="Maximum date (ISO format or 'today')")
    custom_error: Optional[str] = Field(None, description="Custom validation error message")


# 4. FIELD STRUCTURE
class FormField(BaseModel):
    """A single form field - extract text VERBATIM from document."""
    id: Optional[str] = Field(None, description="Section-scoped ID (auto-generated if null)")
    type: Literal["label", "text", "textarea", "number", "date", "checkbox", "checkbox_group", "radio", "select", "signature"] = Field(
        ..., 
        description="Field type: label=read-only, text=single line, textarea=multi-line, checkbox=single tick, checkbox_group=multiple selections, radio=mutually exclusive choice, select=dropdown, date=date picker, number=numeric, signature=signature field"
    )
    label: str = Field(..., description="VERBATIM text from document - copy exactly, do NOT summarize. Include all bullet points with \\n separators.")
    value: Optional[str] = Field(None, description="Pre-filled value if any, otherwise null")
    options: Optional[List[str]] = Field(None, description="Options for radio/checkbox_group/select fields")
    required: bool = Field(default=False, description="True for input fields, False for labels")
    editable: bool = Field(default=True, description="True if empty/needs input, False if pre-filled")
    placeholder: Optional[str] = Field(None, description="Semantic placeholder derived from label context (e.g., 'Enter full name')")
    
    # Conditional logic
    depends_on: Optional[str] = Field(
        None, 
        description="Field ID this field depends on for visibility (e.g., 's1_field_3')"
    )
    show_when: Optional[str] = Field(
        None,
        description="Value condition to show this field (e.g., 'Yes', 'Other', 'not_empty')"
    )
    
    # Visual grouping
    group: Optional[str] = Field(
        None,
        description="Group ID for fields that render together (e.g., 'address', 'contact')"
    )
    group_layout: Optional[Literal["horizontal", "vertical"]] = Field(
        None,
        description="Layout direction for grouped fields (first field in group sets this)"
    )
    
    # Validation
    validation: Optional[ValidationRule] = Field(
        None,
        description="Validation rules for this field"
    )
    
    # Formatting
    label_format: Optional[Literal["plain", "markdown"]] = Field(
        "plain",
        description="How to render label text (plain or markdown for rich formatting)"
    )


# 5. SECTION STRUCTURE
class FormSection(BaseModel):
    """A document section containing fields and tables."""
    id: Optional[str] = Field(None, description="Section ID (auto-generated if null)")
    title: str = Field(..., description="Section title/heading")
    filled_by: Optional[Literal["user", "assessor", "thirdparty"]] = Field(
        None,
        description="Who fills this section (user=applicant, assessor=evaluator, thirdparty=external). If null, inherits from document-level filled_by"
    )
    instructions: Optional[str] = Field(
        None,
        description="Section-specific instructions (e.g., 'Complete only if under 18', 'Office use only')"
    )
    content: List[Union[FormField, FormTable]] = Field(
        ..., 
        description="Ordered list of fields and tables - mirror document's input structure exactly"
    )


# 6. CHUNK EXTRACTION (for large document processing)
class ChunkExtraction(BaseModel):
    """Sections extracted from a document chunk (used for large documents)."""
    sections: List[FormSection] = Field(..., description="Sections extracted from this chunk")


# 7. MAIN DOCUMENT
class ExtractionQuality(BaseModel):
    """Quality metrics for the extraction process."""
    coverage_percentage: float = Field(..., description="Coverage percentage (extracted/expected * 100)")
    is_complete: bool = Field(..., description="True if coverage >= 95%")
    fields_extracted: int = Field(..., description="Number of fields successfully extracted")
    fields_expected: int = Field(..., description="Expected number of fields from markdown analysis")
    warnings: List[str] = Field(default_factory=list, description="Quality warnings and issues")


class FormDocument(BaseModel):
    """Complete form document - extract ALL content from start to end."""
    metadata: DocumentMetadata = Field(..., description="Header/footer metadata")
    name: str = Field(..., description="Full document name (Organization - Title), no abbreviations unless explicitly in document")
    description: str = Field(..., description="1-2 sentence description of form purpose")
    filled_by: Literal["user", "assessor", "thirdparty", "mixed"] = Field(
        ..., 
        description="Who fills this: user=applicant, assessor=evaluator, thirdparty=external reference, mixed=multiple parties"
    )
    sections: List[FormSection] = Field(..., description="All sections from document, extract everything")
    extraction_quality: Optional[ExtractionQuality] = Field(
        None,
        description="Quality metrics from extraction process (populated after validation)"
    )

class SectionAnchor(BaseModel):
    """Section with validation anchors for gap detection in targeted extraction."""
    
    title: str = Field(..., description="Section title/header exactly as written in document")
    section_number: Optional[str] = Field(None, description="Section numbering if present: '1', '2.1', 'A', 'Section 3', etc.")
    start_anchor: str = Field(..., description="First 50-100 characters of section content (after header)")
    end_anchor: str = Field(..., description="Last 50-100 characters of section content (before next section)")
    estimated_density: Literal["light", "medium", "heavy"] = Field(
        ..., 
        description="Field density: 'light' (<10 fields), 'medium' (10-30), 'heavy' (30+)"
    )
    has_tables: bool = Field(..., description="Whether this section contains any tables")
    estimated_field_count: int = Field(..., description="Rough estimate of total input fields in this section")
    start_index_hint: Optional[int] = Field(None, description="Approximate character offset in full document where section starts (GPS coordinate)")
    end_index_hint: Optional[int] = Field(None, description="Approximate character offset where section ends (enables strict slicing)")


class SectionInfo:
    """Information needed to locate and extract a section (non-Pydantic for performance)."""
    
    def __init__(
        self,
        title: str,
        start_anchor: str,
        end_anchor: str,
        start_index_hint: Optional[int] = None,
        end_index_hint: Optional[int] = None,
        next_section_title: Optional[str] = None
    ):
        self.title = title
        self.start_anchor = start_anchor
        self.end_anchor = end_anchor
        self.start_index_hint = start_index_hint
        self.end_index_hint = end_index_hint
        self.next_section_title = next_section_title
    
    @classmethod
    def from_anchor(cls, anchor: SectionAnchor, next_anchor: Optional[SectionAnchor] = None) -> 'SectionInfo':
        """Create SectionInfo from a SectionAnchor schema object."""
        return cls(
            title=anchor.title,
            start_anchor=anchor.start_anchor,
            end_anchor=anchor.end_anchor,
            start_index_hint=anchor.start_index_hint,
            end_index_hint=anchor.end_index_hint,
            next_section_title=next_anchor.title if next_anchor else None
        )
    
    def __repr__(self) -> str:
        return f"SectionInfo(title='{self.title}', hint={self.start_index_hint}-{self.end_index_hint})"


class DocumentStructure(BaseModel):
    """First-pass structure extraction result for targeted extraction."""
    
    document_title: str = Field(..., description="Main title of the document")
    total_sections: int = Field(..., description="Total number of sections identified")
    sections: List[SectionAnchor] = Field(..., description="All sections in document order with anchors")
    metadata_hints: Optional[Dict[str, str]] = Field(None, description="Metadata found: version, organization, codes, etc.")
    
    class Config:
        extra = "forbid"  # Explicitly forbid additional properties


class BatchContinuityHint(BaseModel):
    """Minimal state tracking for batch boundaries (table splitting, context handoff)."""
    
    prepend_content: Optional[str] = Field(
        None, 
        description="Content to prepend to next batch (e.g., table headers for continuation)"
    )
    open_table_name: Optional[str] = Field(
        None,
        description="Name of table that was split mid-way (e.g., 'Employment History')"
    )
    context_note: Optional[str] = Field(
        None,
        description="Human-readable note about the continuation (for debugging/prompt)"
    )
    last_element_type: Optional[str] = Field(
        None,
        description="Type of last element extracted: 'table', 'field', 'section'"
    )


class StreamingState(BaseModel):
    """State passed between streaming chunks for continuity."""
    
    chunk_index: int = Field(0, description="Current chunk number (0-indexed)")
    active_section_title: Optional[str] = Field(
        None,
        description="Title of section in progress when chunk ended"
    )
    active_section_id: Optional[str] = Field(
        None,
        description="ID of section in progress (for merging)"
    )
    table_in_progress: bool = Field(
        False,
        description="True if we cut mid-table"
    )
    table_name: Optional[str] = Field(
        None,
        description="Name of table that was split"
    )
    table_headers: Optional[List[str]] = Field(
        None,
        description="Column headers to replicate for continuation"
    )
    overlap_text: Optional[str] = Field(
        None,
        description="Last ~200 chars from previous chunk for context"
    )
    total_sections_extracted: int = Field(
        0,
        description="Running count of sections extracted so far"
    )


class BatchResult(BaseModel):
    """Result from a single batch extraction in targeted extraction mode."""
    
    batch_id: int = Field(..., description="Batch identifier (0-indexed)")
    sections_requested: List[str] = Field(..., description="Section titles that were requested for extraction")
    sections_extracted: List[FormSection] = Field(..., description="Extracted FormSection objects")
    finish_reason: str = Field(..., description="API finish reason: 'stop' (complete) or 'length' (truncated)")
    output_tokens_used: int = Field(..., description="Actual output tokens consumed by this batch")
