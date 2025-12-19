import os
import re
import time
import html
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
from schemas import (
    FormDocument, FormSection, ChunkExtraction,
    SectionAnchor, DocumentStructure, BatchResult, BatchContinuityHint,
    DocumentMetadata, ExtractionQuality, SectionInfo, StreamingState
)
from config import ExtractionConfig, get_default_config
from result import ExtractionResult
from prompts import (
    SYSTEM_PROMPT,
    build_targeted_extraction_prompt,
)
from constants import (
    GAP_WARNING_THRESHOLD_CHARS,
    SUBSTANTIVE_GAP_THRESHOLD_CHARS,
    ANCHOR_SEARCH_BUFFER_BEFORE,
    ANCHOR_SEARCH_EXTENDED_WINDOW,
    TABLE_TRUNCATION_THRESHOLD_RATIO,
    EXTRACTION_COMPLETE_THRESHOLD_PCT,
    DUPLICATE_WARNING_THRESHOLD_RATIO,
    ESTIMATED_TOKENS_PER_SECTION_METADATA,
    TOKENS_PER_SECTION_ESTIMATE,
    ESTIMATED_TOKENS_PER_FIELD,
    ESTIMATED_TOKENS_PER_TABLE,
    ESTIMATED_TOKENS_PER_SECTION,
    MODEL_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
    SYSTEM_PROMPT_TOKENS,
    SAFETY_MARGIN_TOKENS,
    AVAILABLE_INPUT_TOKENS,
    SAFE_OUTPUT_TOKENS,
    MAX_SECTIONS_PER_BATCH,
    SUPER_CHUNK_THRESHOLD_TOKENS,
    RATE_LIMIT_RETRY_BASE_DELAY,
    CHUNK_THRESHOLD_TOKENS,
    CHUNK_MAX_SIZE_TOKENS,
    CHUNK_MIN_SIZE_TOKENS,
    CHUNK_MAX_SIZE,
    CHUNK_MIN_SIZE,
    OVERLAP_SIZE,
)

load_dotenv(Path(__file__).parent / ".env")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5")

# Initialize tiktoken encoder
try:
    TOKENIZER = tiktoken.get_encoding("o200k_base")
except Exception:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")

DEFAULT_CONFIG = get_default_config()


def get_model_limits(config: 'ExtractionConfig') -> dict:
    """Get model-specific limits from config."""
    return {
        'context_tokens': config.model.context_tokens,
        'max_output_tokens': config.model.max_output_tokens,
        'available_input': config.available_input_tokens,
        'safe_output': config.safe_output_tokens,
    }


def estimate_tokens(text: str) -> int:
    """Accurately count tokens using tiktoken."""
    return len(TOKENIZER.encode(text))


def _clean_markdown(text: str) -> str:

    lines = text.split('\n')
    cleaned_lines = []
    
    # Patterns for noise lines to remove
    noise_patterns = [
        # Page numbers: "Page 1 of 50", "Page 1", "- 1 -", "1 / 50"
        re.compile(r'^\s*Page\s+\d+(\s+(of|/)\s+\d+)?\s*$', re.IGNORECASE),
        re.compile(r'^\s*-?\s*\d+\s*-?\s*$'),  # Standalone numbers like "- 1 -" or "1"
        re.compile(r'^\s*\d+\s*/\s*\d+\s*$'),  # "1 / 50"
        
        # Common footer text
        re.compile(r'^\s*Confidential\s*[-–—]?\s*(Internal Use Only)?\s*$', re.IGNORECASE),
        re.compile(r'^\s*©\s*\d{4}.*$'),  # Copyright lines
        re.compile(r'^\s*All Rights Reserved\.?\s*$', re.IGNORECASE),
        
        # Version/revision stamps repeated
        re.compile(r'^\s*Version\s+[\d.]+\s*$', re.IGNORECASE),
        re.compile(r'^\s*Rev\.?\s+[\d.]+\s*$', re.IGNORECASE),
        
        # Empty table separators (multiple |---|)
        re.compile(r'^\s*(\|\s*[-:]+\s*)+\|\s*$'),  # Keep this for actual tables
    ]
    
    # Track repeated lines (headers/footers appear multiple times)
    line_counts: Dict[str, int] = {}
    for line in lines:
        normalized = line.strip().lower()
        if len(normalized) > 5:  # Ignore very short lines
            line_counts[normalized] = line_counts.get(normalized, 0) + 1
    
    # Lines appearing 3+ times are likely headers/footers
    repeated_lines = {k for k, v in line_counts.items() if v >= 3}
    
    for line in lines:
        # Skip empty lines (we'll keep some for structure)
        if not line.strip():
            cleaned_lines.append(line)
            continue
        
        # Check against noise patterns
        is_noise = False
        for pattern in noise_patterns:
            if pattern.match(line):
                is_noise = True
                break
        
        # Check if it's a repeated header/footer
        normalized = line.strip().lower()
        if normalized in repeated_lines and len(normalized) < 100:
            is_noise = True
        
        if not is_noise:
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines)
    
    # Remove excessive blank lines (more than 2 consecutive)
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    
    return result

class ClientManager:
    _sync_client: Optional[OpenAI] = None
    
    @classmethod
    def get_sync_client(cls) -> OpenAI:
        """Get or create sync OpenAI client."""
        if cls._sync_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            # Increase timeout for large document processing
            cls._sync_client = OpenAI(
                api_key=api_key,
                timeout=2500.0, 
                max_retries=2,  # Built-in retries
            )
        return cls._sync_client
    
    @classmethod
    def reset(cls) -> None:
        """Reset clients (useful for testing or API key rotation)."""
        if cls._sync_client:
            cls._sync_client.close()
            cls._sync_client = None


def get_client() -> OpenAI:
    """Get sync OpenAI client (convenience function)."""
    return ClientManager.get_sync_client()


@dataclass
class SemanticUnit:
    """A complete semantic unit (section header + all its content)."""
    header: str  
    content: str  
    start_pos: int  
    end_pos: int  
    contains_table: bool = False  
    
    @property
    def full_text(self) -> str:
        return f"{self.header}\n{self.content}".strip()
    
    @property
    def size(self) -> int:
        return len(self.full_text)


@dataclass
class DocumentStrategy:
    """Smart traffic cop that determines the optimal extraction strategy."""
    mode: str  # "single_pass", "standard_chunking", or "deep_chunking"
    utilization_ratio: float
    should_chunk_structure: bool
    recommended_structure_chunk_size: int
    
    @staticmethod
    def determine(doc_tokens: int, model_context: int, max_output: int) -> 'DocumentStrategy':
        utilization_ratio = doc_tokens / model_context
        
        # Conservative output estimate: ~1.5x input tokens for JSON structure
        estimated_output = doc_tokens * 1.5

        SINGLE_PASS_OUTPUT_LIMIT = max_output * 0.8  # 80% of 128k = ~102k tokens

        if estimated_output < SINGLE_PASS_OUTPUT_LIMIT:
            return DocumentStrategy(
                mode="single_pass",
                utilization_ratio=utilization_ratio,
                should_chunk_structure=False,
                recommended_structure_chunk_size=model_context
            )
        
        elif utilization_ratio < 3.0:
            # Standard chunking zone
            return DocumentStrategy(
                mode="standard_chunking",
                utilization_ratio=utilization_ratio,
                should_chunk_structure=True,
                recommended_structure_chunk_size=int(model_context * 0.4)
            )
        
        else:
            # Deep chunking zone: Massive documents (4MB+)
            return DocumentStrategy(
                mode="deep_chunking",
                utilization_ratio=utilization_ratio,
                should_chunk_structure=True,
                recommended_structure_chunk_size=int(model_context * 0.3)
            )

def should_chunk(markdown: str) -> bool:
    """Determine if document needs chunking based on actual token count."""
    token_count = estimate_tokens(markdown)
    return token_count > CHUNK_THRESHOLD_TOKENS


def split_into_chunks(markdown: str) -> List[str]:

    if estimate_tokens(markdown) <= CHUNK_MAX_SIZE_TOKENS:
        return [markdown]
    
    # Step 1: Parse into semantic units ONCE - preserve metadata
    units = _parse_semantic_units(markdown)
    
    # Step 2: Greedily combine while respecting semantic boundaries
    chunks = _semantic_greedy_combine(units)
    
    # Step 3: Final safety check for any remaining oversized chunks
    chunks = _subdivide_large_chunks(chunks, _hard_split)
    
    # Step 4: Validate chunks for orphan content
    chunks = _validate_chunks(chunks)
    
    return chunks


def _parse_semantic_units(markdown: str) -> List[SemanticUnit]:

    units = []
    
    # Find all header positions (Level 1, 2, or 3: #, ##, ###)
    header_pattern = r'^(#{1,3}\s+.+)$'
    matches = list(re.finditer(header_pattern, markdown, re.MULTILINE))
    
    if not matches:
        # No headers found - treat entire document as one unit
        return [SemanticUnit(
            header="",
            content=markdown,
            start_pos=0,
            end_pos=len(markdown),
            contains_table=_contains_table(markdown)
        )]
    
    # Extract units between headers
    for i, match in enumerate(matches):
        header = match.group(1)
        header_start = match.start()
        
        # Content goes from after this header to the next header (or end)
        content_start = match.end() + 1
        if i + 1 < len(matches):
            content_end = matches[i + 1].start()
        else:
            content_end = len(markdown)
        
        content = markdown[content_start:content_end].strip()
        
        unit = SemanticUnit(
            header=header,
            content=content,
            start_pos=header_start,
            end_pos=content_end,
            contains_table=_contains_table(content)
        )
        units.append(unit)
    
    # Handle content before first header (preamble)
    if matches[0].start() > 0:
        preamble = markdown[0:matches[0].start()].strip()
        if preamble:
            units.insert(0, SemanticUnit(
                header="",
                content=preamble,
                start_pos=0,
                end_pos=matches[0].start(),
                contains_table=_contains_table(preamble)
            ))
    
    return units

def _contains_table(text: str) -> bool:
    """Check if text contains a markdown table."""
    # Look for table separators like |---|---|
    return bool(re.search(r'\|\s*:?-+:?\s*\|', text))

def _find_table_boundaries(text: str) -> List[Tuple[int, int]]:

    boundaries = []
    lines = text.split('\n')
    
    in_table = False
    table_start = 0
    
    for i, line in enumerate(lines):
        # Detect table separator line
        if re.match(r'^\|\s*:?-+:?\s*\|', line):
            if not in_table:
                # Table starts at previous line (header row)
                table_start = max(0, i - 1)
                in_table = True
        elif in_table:
            # Check if still in table (lines must start with |)
            if not line.strip().startswith('|'):
                # Table ended
                boundaries.append((table_start, i - 1))
                in_table = False
    
    # Handle table that extends to end
    if in_table:
        boundaries.append((table_start, len(lines) - 1))
    
    return boundaries

def _semantic_greedy_combine(units: List[SemanticUnit]) -> List[str]:

    chunks = []
    current_chunk_units = []
    current_chunk_size = 0
    
    for unit in units:
        unit_size = unit.size
        potential_size = current_chunk_size + unit_size + 2  # +2 for \n\n separator
        
        # Special case: single unit is too large
        if unit_size > CHUNK_MAX_SIZE:
            # Flush current chunk
            if current_chunk_units:
                chunks.append(_combine_units(current_chunk_units))
                current_chunk_units = []
                current_chunk_size = 0
            
            # Try to split this unit intelligently
            split_chunks = _split_large_unit(unit)
            chunks.extend(split_chunks)
            continue
        
        # Decision: add to current chunk or start new one?
        should_add = (
            potential_size <= CHUNK_MAX_SIZE or  # It fits
            current_chunk_size < CHUNK_MIN_SIZE   # Current too small
        )
        
        if should_add:
            current_chunk_units.append(unit)
            current_chunk_size = potential_size
        else:
            # Finalize current chunk and start new one
            if current_chunk_units:
                chunks.append(_combine_units(current_chunk_units))
            current_chunk_units = [unit]
            current_chunk_size = unit_size
    
    # Don't forget the last chunk
    if current_chunk_units:
        chunks.append(_combine_units(current_chunk_units))
    
    return chunks if chunks else [units[0].full_text if units else ""]


def _combine_units(units: List[SemanticUnit]) -> str:
    """Combine semantic units into a single chunk."""
    return "\n\n".join(unit.full_text for unit in units)


def _split_large_unit(unit: SemanticUnit, max_size: int = None) -> List[str]:

    if max_size is None:
        max_size = CHUNK_MAX_SIZE
    
    text = unit.full_text
    
    # If unit contains table, try to split around it
    if unit.contains_table:
        return _split_around_tables(text, max_size)
    
    # Otherwise use paragraph splitting
    para_chunks = _split_on_paragraphs(text, max_size)
    
    # If still too large, hard split
    if any(len(chunk) > max_size for chunk in para_chunks):
        final_chunks = []
        for chunk in para_chunks:
            if len(chunk) > max_size:
                final_chunks.extend(_hard_split(chunk, max_size))
            else:
                final_chunks.append(chunk)
        return final_chunks
    
    return para_chunks


def _split_around_tables(text: str, max_size: int = None) -> List[str]:

    if max_size is None:
        max_size = CHUNK_MAX_SIZE
    
    min_size = int(max_size * 0.3)  # 30% of max_size
    
    lines = text.split('\n')
    table_boundaries = _find_table_boundaries(text)
    
    if not table_boundaries:
        # No tables found despite flag - fall back to paragraph split
        return _split_on_paragraphs(text, max_size)
    
    chunks = []
    current_chunk_lines = []
    line_idx = 0
    
    for table_start, table_end in table_boundaries:
        # Add lines before table to current chunk
        while line_idx < table_start:
            current_chunk_lines.append(lines[line_idx])
            line_idx += 1
        
        # Check if adding table would exceed limit
        current_text = '\n'.join(current_chunk_lines)
        table_lines = lines[table_start:table_end + 1]
        table_text = '\n'.join(table_lines)
        
        potential_size = len(current_text) + len(table_text) + 2
        
        if potential_size > max_size and len(current_text) >= min_size:
            # Finalize chunk before table
            if current_chunk_lines:
                chunks.append('\n'.join(current_chunk_lines))
            current_chunk_lines = []
        
        # Add table to current chunk
        current_chunk_lines.extend(table_lines)
        line_idx = table_end + 1
        
        # If table itself is huge, flush it as its own chunk
        if len(table_text) > min_size:
            chunks.append('\n'.join(current_chunk_lines))
            current_chunk_lines = []
    
    # Add remaining lines
    while line_idx < len(lines):
        current_chunk_lines.append(lines[line_idx])
        line_idx += 1
    
    if current_chunk_lines:
        chunks.append('\n'.join(current_chunk_lines))
    
    return chunks if chunks else [text]


def _split_on_paragraphs(text: str, max_size: int = None) -> List[str]:

    if max_size is None:
        max_size = CHUNK_MAX_SIZE
    
    min_size = int(max_size * 0.3)  # 30% of max_size
    
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        potential_size = len(current_chunk) + len(para) + 2
        
        
        should_add = (
            potential_size <= max_size or
            len(current_chunk) < min_size
        )
        
        if should_add:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]


def _hard_split(text: str, max_size: int = None) -> List[str]:

    if max_size is None:
        max_size = CHUNK_MAX_SIZE
    
    overlap = min(OVERLAP_SIZE, int(max_size * 0.05))  # 5% overlap or OVERLAP_SIZE
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_size, len(text))
        
        if end == len(text):
            # Last chunk 
            remaining = text[start:]
            
            if len(remaining) < int(max_size * 0.3) and chunks:
                chunks[-1] = chunks[-1] + remaining
            else:
                chunks.append(remaining)
            break
        
        
        search_start = start + int(max_size * 0.8)
        break_point = text.rfind('\n', search_start, end)
        if break_point == -1:
            break_point = text.rfind(' ', search_start, end)
        if break_point == -1 or break_point <= start:
            break_point = end
        
        chunks.append(text[start:break_point])

        start = max(start + 1, break_point - overlap)
    
    return chunks


def _subdivide_large_chunks(chunks: List[str], split_func) -> List[str]:
    """Apply split function to any chunks that exceed max size."""
    result = []
    for chunk in chunks:
        if len(chunk) > CHUNK_MAX_SIZE:
            result.extend(split_func(chunk))
        else:
            result.append(chunk)
    return result


def _validate_chunks(chunks: List[str]) -> List[str]:
    """Ensure each chunk (except first) has context markers if starting without header."""
    if len(chunks) <= 1:
        return chunks
    
    validated = [chunks[0]] 
    
    for i, chunk in enumerate(chunks[1:], 1):
        stripped = chunk.strip()
        
        # Check if chunk starts with a proper header
        has_header = stripped.startswith('#')
        
        if not has_header:
            # Mark as continuation to help LLM understand context
            validated.append(f"[CONTINUATION OF PREVIOUS SECTION]\n\n{chunk}")
        else:
            validated.append(chunk)
    
    return validated

def _is_continuation_section(section: FormSection) -> bool:
    """Check if section appears to be a continuation (no full structure)."""
  
    if not section.content:
        return False
    
    # If first item is a label with long text, likely a proper section start
    first_item = section.content[0]
    if hasattr(first_item, 'type') and first_item.type == 'label':
        if hasattr(first_item, 'label') and len(first_item.label) > 100:
            return False  # Proper section intro
    
    return True  # Likely a continuation


def _renumber_content_ids(section: FormSection, section_num: int) -> None:
    """Renumber content IDs within a section."""
    for j, item in enumerate(section.content):
        if hasattr(item, 'id'):
            if hasattr(item, 'type') and item.type == 'table':
                item.id = f"s{section_num}_table_{j + 1}"
            else:
                item.id = f"s{section_num}_field_{j + 1}"


def estimate_output_tokens(markdown: str) -> int:

    # Count checkbox symbols
    checkbox_count = len(re.findall(r'[☐☑□■]', markdown))
    bracket_checkbox_count = len(re.findall(r'\[\s*[xX]?\s*\]', markdown))
    
    # Count blank input patterns
    blank_lines = len(re.findall(r'_{3,}|\.{3,}|\[_{2,}\]', markdown))
    
    # Count table rows (exclude header separators)
    table_rows = len(re.findall(r'^\|[^-].*\|$', markdown, re.MULTILINE))
    
    # Count section headers
    section_count = len(re.findall(r'^#{1,3}\s+', markdown, re.MULTILINE))
    
    # Count signature/date fields
    signature_fields = len(re.findall(
        r'(?i)(signature|sign here|date:|signed by|authorized)', 
        markdown
    ))
    
    # Calculate estimated fields
    estimated_fields = (
        checkbox_count + 
        bracket_checkbox_count + 
        blank_lines + 
        signature_fields +
        int(table_rows * 0.5)  # Tables generate fewer individual fields
    )
    
    # Calculate estimated output tokens
    base_output = (
        (estimated_fields * ESTIMATED_TOKENS_PER_FIELD) +
        (section_count * ESTIMATED_TOKENS_PER_SECTION) +
        500  # Document metadata overhead
    )

    reasoning_safety_factor = 1.2  
    estimated_output = int(base_output * reasoning_safety_factor)
    
    return estimated_output


def _normalize_for_search(text: str) -> str:
    """Normalize text for fuzzy matching."""
    # Unescape HTML entities (e.g. &amp; -> &)
    text = html.unescape(text)
    # Remove markdown symbols
    text = re.sub(r'[#*\-_]+', ' ', text)
    # Collapse multiple spaces/newlines into single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def _extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases (3+ word sequences) from text for matching."""
    words = _normalize_for_search(text).split()
    phrases = []
    # Extract 3-5 word phrases
    for length in [5, 4, 3]:
        for i in range(len(words) - length + 1):
            phrase = ' '.join(words[i:i+length])
            if len(phrase) > 10:  # Only meaningful phrases
                phrases.append(phrase)
    return phrases


def _fuzzy_find(haystack: str, needle: str, start: int = 0) -> int:
    """Find needle in haystack with multiple fallback strategies."""
    search_text = haystack[start:]
    norm_haystack = _normalize_for_search(search_text)
    norm_needle = _normalize_for_search(needle)
    
    # Strategy 1: Exact normalized match
    norm_pos = norm_haystack.find(norm_needle)
    if norm_pos != -1:
        return start + norm_pos
    
    # Strategy 2: First 60% of needle (handles end variations)
    if len(norm_needle) > 20:
        partial = norm_needle[:int(len(norm_needle) * 0.6)]
        norm_pos = norm_haystack.find(partial)
        if norm_pos != -1:
            return start + norm_pos
    
    # Strategy 3: Last 60% of needle (handles start variations)
    if len(norm_needle) > 20:
        partial = norm_needle[int(len(norm_needle) * 0.4):]
        norm_pos = norm_haystack.find(partial)
        if norm_pos != -1:
            return start + norm_pos
    
    # Strategy 4: Key phrase matching (most flexible)
    key_phrases = _extract_key_phrases(needle)
    for phrase in key_phrases:
        pos = norm_haystack.find(phrase)
        if pos != -1:
            return start + pos
    
    return -1

def _extract_table_header(markdown: str, table_start_pos: int) -> Optional[str]:

    lines = markdown[table_start_pos:].split('\n')
    
    header_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # First line should be header
        if i == 0 and stripped.startswith('|') and stripped.endswith('|'):
            header_lines.append(line)
        # Second line should be separator
        elif i == 1 and re.match(r'^\|\s*:?-+:?\s*\|', stripped):
            header_lines.append(line)
            break
        else:
            break
    
    if len(header_lines) >= 2:
        return '\n'.join(header_lines) + '\n'
    return None

def _find_last_complete_table_row(markdown: str, start_pos: int, limit_pos: int) -> int:

    text_segment = markdown[start_pos:limit_pos]
    lines = text_segment.split('\n')
    table_row_pattern = re.compile(r'^\s*\|.*\|\s*$')
    last_row_end = start_pos 
    current_pos = start_pos
    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        
        if table_row_pattern.match(line):
            # This is a complete table row
            last_row_end = current_pos + line_len
        elif line.strip() and not re.match(r'^\|\s*:?-+:?\s*\|', line):
            # Hit non-table content, stop
            break
        
        current_pos += line_len
    
    return last_row_end if last_row_end > start_pos else limit_pos

def _extend_to_table_boundary(
    markdown: str, 
    end_pos: int,
    max_extension: int = 50000
) -> Tuple[int, Optional[str], Optional[str]]:
   
    # Look ahead to see if we're in a table
    peek_ahead = markdown[end_pos:end_pos + 500]
    lines_after = peek_ahead.split('\n')
    table_row_pattern = re.compile(r'^\s*\|.*\|\s*$')
    next_line = lines_after[0] if lines_after else ""
    is_in_table = table_row_pattern.match(next_line.strip())
    
    prev_newline = markdown.rfind('\n', max(0, end_pos - 200), end_pos)
    current_line = markdown[prev_newline:end_pos].strip() if prev_newline != -1 else ""
    is_in_table = is_in_table or table_row_pattern.match(current_line)
    
    if not is_in_table:
        # Not in a table, no extension needed
        return end_pos, None, None
    
    scan_pos = end_pos
    extended_end = end_pos
    max_scan = min(end_pos + max_extension, len(markdown))
    
    table_start = end_pos
    for i in range(end_pos - 1, max(0, end_pos - 5000), -1):
        if markdown[i] == '\n':
            next_line_start = i + 1
            next_line_end = markdown.find('\n', next_line_start)
            if next_line_end == -1:
                next_line_end = len(markdown)
            
            line_content = markdown[next_line_start:next_line_end].strip()
            if not table_row_pattern.match(line_content) and line_content:
                # Found start of table (line before first row)
                table_start = i + 1
                break
    
    # Extract table header for potential replication
    table_header = _extract_table_header(markdown, table_start)
    
    # Try to find table name (look for header before table)
    table_name = None
    pre_table_text = markdown[max(0, table_start - 200):table_start]
    title_match = re.search(r'(#{1,3}\s+(.+)|([A-Z][^.\n]{3,50}):?)\s*$', pre_table_text)
    if title_match:
        table_name = title_match.group(2) or title_match.group(3)
        if table_name:
            table_name = table_name.strip()
    
    # Scan forward to find end of table
    remaining = markdown[scan_pos:max_scan]
    remaining_lines = remaining.split('\n')
    
    for line in remaining_lines:
        if table_row_pattern.match(line.strip()):
            extended_end = scan_pos + len(line) + 1
            scan_pos = extended_end
        elif line.strip() and not re.match(r'^\|\s*:?-+:?\s*\|', line):
            # Hit non-table content
            break
        else:
            # Empty line or separator, continue
            scan_pos += len(line) + 1
    
    if extended_end > end_pos + 50:

        if extended_end >= max_scan - 100:

            last_complete_row = _find_last_complete_table_row(markdown, table_start, extended_end)
            # FIX: Don't require table_name - use generic if not found
            safe_table_name = table_name or "Continued Table"
            if table_header:
                return last_complete_row, table_header, safe_table_name
            else:
                return last_complete_row, None, None
        else:
            # Table ended naturally within our extension
            return extended_end, None, None
    
    # Minimal or no extension needed
    return end_pos, None, None

def _extract_section_content_by_anchors(
    markdown: str, 
    start_anchor: str, 
    end_anchor: str,
    start_index_hint: Optional[int] = None,
    title_hint: Optional[str] = None,
    next_section_title: Optional[str] = None
) -> str:
    
    search_start = 0
    if start_index_hint is not None:
        # Buffer: Start searching 5k chars BEFORE the hint to be safe
        search_start = max(0, start_index_hint - ANCHOR_SEARCH_BUFFER_BEFORE)
    
    # Find start position (search from GPS coordinate, not from beginning)
    start_pos = markdown.find(start_anchor, search_start)
    if start_pos == -1:
        # Fallback 1: Case-insensitive
        start_pos = markdown.lower().find(start_anchor.lower(), search_start)
        if start_pos == -1:
            # Fallback 2: Fuzzy/normalized search from hint
            start_pos = _fuzzy_find(markdown, start_anchor, search_start)
            
            if start_pos == -1 and len(start_anchor) > 30:
              
                sub_anchor = start_anchor[:int(len(start_anchor) * 0.6)]
                start_pos = _fuzzy_find(markdown, sub_anchor, search_start)

            if start_pos == -1:
             
                if start_index_hint is not None:
                    # Search ±20k window around hint to catch badly offset anchors
                    window_start = max(0, start_index_hint - ANCHOR_SEARCH_EXTENDED_WINDOW)
                    window_end = min(len(markdown), start_index_hint + ANCHOR_SEARCH_EXTENDED_WINDOW)
                    start_pos = _fuzzy_find(markdown[window_start:window_end], start_anchor, 0)
                    if start_pos != -1:
                        start_pos += window_start  # Adjust to absolute position
                
                # Final fallback: Full document search (risky with repetitive headers)
                if start_pos == -1:
                    start_pos = _fuzzy_find(markdown, start_anchor, 0)
                    
                    # Fallback 4: Title Search (The "Hail Mary" pass)
                    if start_pos == -1 and title_hint:
                        # Normalize title - remove markdown symbols
                        clean_title = title_hint
                        clean_title = re.sub(r'^#+\s*', '', clean_title)  # Remove leading ##
                        clean_title = re.sub(r'\*+', '', clean_title)  # Remove ** and *
                        clean_title = re.sub(r'_+', '', clean_title)  # Remove __ and _
                        clean_title = re.sub(r'[^\w\s\-\&\:\(\)]', '', clean_title).strip()
                        
                        # Try multiple title patterns
                        title_patterns = [
                            title_hint,  # Exact title
                            f"## {title_hint}",  # With ## header
                            f"### {title_hint}",  # With ### header
                            f"# {title_hint}",  # With # header
                            clean_title,  # Cleaned title
                        ]
                        
                        # Also try extracting key words (first 3-4 significant words)
                        title_words = [w for w in clean_title.split() if len(w) > 3]
                        if len(title_words) >= 3:
                            title_patterns.append(' '.join(title_words[:4]))
                        
                        title_pos = -1
                        for pattern in title_patterns:
                            # Try exact match
                            title_pos = markdown.find(pattern, search_start)
                            if title_pos != -1:
                                break
                            
                            # Try case-insensitive
                            title_pos = markdown.lower().find(pattern.lower(), search_start)
                            if title_pos != -1:
                                break
                            
                            # Try fuzzy
                            title_pos = _fuzzy_find(markdown, pattern, search_start)
                            if title_pos != -1:
                                break
                        
                        if title_pos != -1:
                            # Found the title! Content starts after the title line
                            newline_pos = markdown.find('\n', title_pos)
                            if newline_pos != -1:
                                start_pos = newline_pos + 1
                            else:
                                start_pos = title_pos + len(title_hint)
                                
                    if start_pos == -1:
                        return ""
    
    # Find end position (search after start)
    search_min = start_pos + len(start_anchor)
    end_pos = markdown.find(end_anchor, search_min)
    if end_pos == -1:
        # Fallback 1: Case-insensitive
        end_pos = markdown.lower().find(end_anchor.lower(), search_min)
        if end_pos == -1:
            # Fallback 2: Fuzzy search
            end_pos = _fuzzy_find(markdown, end_anchor, search_min)
            
            if end_pos == -1 and len(end_anchor) > 30:
                # Fallback 2.5: Substring search (last 60% of anchor)
                sub_anchor = end_anchor[int(len(end_anchor) * 0.4):]
                end_pos = _fuzzy_find(markdown, sub_anchor, search_min)
                if end_pos != -1:
                    end_pos += len(sub_anchor) # Approximate end

            if end_pos == -1:
                # End anchor not found - try to find next section title
                found_next = False
                if next_section_title:
                    # Try to find next section title
                    # Normalize title - remove markdown symbols
                    clean_next_title = next_section_title
                    clean_next_title = re.sub(r'^#+\s*', '', clean_next_title)
                    clean_next_title = re.sub(r'\*+', '', clean_next_title)
                    clean_next_title = re.sub(r'_+', '', clean_next_title)
                    clean_next_title = re.sub(r'[^\w\s\-\&\:\(\)]', '', clean_next_title).strip()
                    
                    # Try fuzzy find for next title
                    next_pos = _fuzzy_find(markdown, clean_next_title, search_min)
                    if next_pos != -1:
                        end_pos = next_pos
                        found_next = True
                
                if not found_next:
                   
                    next_section = re.search(r'\n#{1,3}\s+', markdown[start_pos + 100:])
                    if next_section:
                        end_pos = start_pos + 100 + next_section.start()
                    else:
                        end_pos = len(markdown)
            else:
                end_pos += len(end_anchor)  # Approximate
        else:
            end_pos += len(end_anchor)
    else:
        end_pos += len(end_anchor)
    

    extended_end, _, _ = _extend_to_table_boundary(markdown, end_pos)
    end_pos = extended_end
    
    return markdown[start_pos:end_pos].strip()


def _extract_structure_regex(markdown: str, min_header_level: int = 2) -> DocumentStructure:
   
    sections = []
    matches = []
    
    # Strategy 1: 
    header_pattern = re.compile(rf'^(#{{1,{min_header_level}}})(?!#)\s+(.+?)$', re.MULTILINE)
    md_matches = list(header_pattern.finditer(markdown))
    
    # Filter out invalid matches
    valid_md_matches = []
    for match in md_matches:
        title = match.group(2).strip()
        if not title or len(title) < 3 or re.match(r'^[#\-_\*\s]+$', title):
            continue
        valid_md_matches.append(match)
    md_matches = valid_md_matches
    
  
    doc_tokens = estimate_tokens(markdown)
    tokens_per_section = doc_tokens / max(1, len(md_matches)) if md_matches else doc_tokens
    
    # If we have enough headers (avg <30k tokens each), use them
    if md_matches and tokens_per_section < 30_000:
        matches = md_matches
        print(f"[Extractor]   Found {len(matches)} markdown headers")
    else:
        # Strategy 2: Aggressive header detection for documents without/few markdown headers
        if md_matches:
            print(f"[Extractor]   Found {len(md_matches)} markdown header(s) but sections too large ({tokens_per_section:,.0f} tokens each)")
        print(f"[Extractor]   Using aggressive header detection...")
        
        # Pattern 1: Bold headers **HEADER TEXT** or **Header Text** (relaxed)
        bold_pattern = re.compile(r'^\*\*([A-Za-z][^\n*]{2,100}?)\*\*\s*$', re.MULTILINE)
        
        # Pattern 2: All-caps headers (at least 8 chars, mostly uppercase)
        allcaps_pattern = re.compile(r'^([A-Z][A-Z0-9\s\-:&\(\)]{7,100})$', re.MULTILINE)
        
        # Pattern 3: Underlined headers <u>Header</u>
        underline_pattern = re.compile(r'^<u>([^<]{3,100})</u>\s*$', re.MULTILINE)
        
        # Pattern 4: Numbered sections (e.g., "1. SECTION NAME", "PART A", "Section 1:")
        numbered_pattern = re.compile(r'^(?:(?:PART|SECTION|Section|Part|UNIT|Unit|CHAPTER|Chapter)\s+[A-Z0-9]+[:\-\s]*|[0-9]+\.)\s*([A-Za-z][^\n]{3,100})$', re.MULTILINE)
        
        # Pattern 5: Colon-ended headers (common in forms: "APPLICANT DETAILS:")
        colon_pattern = re.compile(r'^([A-Z][A-Z0-9\s\-&]{5,60}):\s*$', re.MULTILINE)
        
        # Pattern 6: Page/Section markers commonly found in converted docs
        page_section_pattern = re.compile(r'^(?:Page\s+\d+\s*[-–—]\s*)?([A-Z][A-Za-z\s\-&]{5,60})\s*$', re.MULTILINE)
        
        # Collect all potential headers with their positions
        potential_headers = []
        
        for pattern_name, pattern in [
            ('bold', bold_pattern),
            ('allcaps', allcaps_pattern), 
            ('underline', underline_pattern),
            ('numbered', numbered_pattern),
            ('colon', colon_pattern),
            ('page_section', page_section_pattern)
        ]:
            for match in pattern.finditer(markdown):
                title = match.group(1).strip()
                
                # Validation: Skip if too short
                if len(title) < 3:
                    continue
                if re.match(r'^[#\-_\*\s\|:=]+$', title):
                    continue
                if title.count('|') > 2:  
                    continue
                
                # Additional validation: Must have at least one word with 3+ chars
                words = [w for w in re.findall(r'\w+', title) if len(w) >= 3]
                if not words:
                    continue
                
                potential_headers.append({
                    'match': match,
                    'title': title,
                    'pos': match.start(),
                    'type': pattern_name
                })
        
        # Sort by position
        potential_headers.sort(key=lambda x: x['pos'])
        
        # Remove duplicates (headers detected by multiple patterns)
        unique_headers = []
        last_pos = -500
        for header in potential_headers:
            if header['pos'] - last_pos > 50:  # Must be at least 50 chars apart
                unique_headers.append(header)
                last_pos = header['pos']
        
        if unique_headers:
            print(f"[Extractor]   Found {len(unique_headers)} headers via aggressive detection")
            # Convert to match-like objects for consistency
            for header in unique_headers:
                matches.append(header['match'])
            

            avg_tokens_per_section = doc_tokens / len(unique_headers)
            if avg_tokens_per_section > 30_000:
                print(f"[Extractor]     Sections still too large ({avg_tokens_per_section:,.0f} tokens avg), will sub-chunk")
        else:
            # Strategy 3: FAIL-SAFE - No headers detected at all
            # Split document into chunks by size to prevent single-batch overload
            print(f"[Extractor]     No headers detected - using fail-safe chunking")
            
            doc_tokens = estimate_tokens(markdown)
            max_chunk_tokens = 30_000  # ~30k tokens per chunk for better extraction
            
            if doc_tokens <= max_chunk_tokens:
                # Small enough to treat as one section
                content = markdown.strip()
                sections.append(SectionAnchor(
                    title="Document Content",
                    section_number=None,
                    start_anchor=content[:100] if content else "",
                    end_anchor=content[-100:] if content else "",
                    estimated_density="heavy",
                    has_tables='|' in content,
                    estimated_field_count=len(content) // 50,
                    start_index_hint=0
                ))
            else:
                # Split into semantic chunks
                print(f"[Extractor]   Splitting large document ({doc_tokens:,} tokens) into chunks...")
                chunk_size = int(len(markdown) / (doc_tokens / max_chunk_tokens))
                
                num_chunks = max(2, doc_tokens // max_chunk_tokens + 1)
                chunk_size = len(markdown) // num_chunks
                
                for i in range(num_chunks):
                    start_pos = i * chunk_size
                    end_pos = start_pos + chunk_size if i < num_chunks - 1 else len(markdown)
                    
                   
                    if i < num_chunks - 1:
                        break_point = markdown.find('\n\n', end_pos - 500, end_pos + 500)
                        if break_point != -1:
                            end_pos = break_point
                    
                    content = markdown[start_pos:end_pos].strip()
                    
                    # Estimate fields in this chunk
                    checkbox_count = len(re.findall(r'[☐☑□■\[\]]', content))
                    blank_count = len(re.findall(r'_{3,}|\.{5,}', content))
                    table_rows = len(re.findall(r'^\|[^-].*\|$', content, re.MULTILINE))
                    estimated_fields = checkbox_count + blank_count + int(table_rows * 0.5)
                    
                    if estimated_fields == 0:
                        estimated_fields = len(content) // 100
                    
                    sections.append(SectionAnchor(
                        title=f"Section {i+1}",
                        section_number=str(i+1),
                        start_anchor=content[:100] if content else "",
                        end_anchor=content[-100:] if content else "",
                        estimated_density="heavy",
                        has_tables='|' in content and '---' in content,
                        estimated_field_count=estimated_fields,
                        start_index_hint=start_pos
                    ))
                
                print(f"[Extractor]   Created {len(sections)} fail-safe chunks")
            
            return DocumentStructure(
                document_title="Document (Auto-chunked)",
                total_sections=len(sections),
                sections=sections
            )
    
    # Process detected headers 
    doc_title = None
    for i, match in enumerate(matches):
        # Extract title based on match type
        if hasattr(match, 'group'):
            # Markdown header match
            if match.lastindex >= 2:
                title = match.group(2).strip()
            else:
                title = match.group(1).strip()
        else:
            title = "Section"
        
        if i == 0:
            doc_title = title
        
        start_pos = match.end() + 1
        
       
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start() - 1
        else:
            end_pos = len(markdown)
        
        raw_content = markdown[start_pos:end_pos]
        content = raw_content.strip()
        
        # Calculate how many chars were stripped from the start
        if content:
            strip_offset = raw_content.find(content[0]) if content else 0
        else:
            strip_offset = 0
        
        # Adjusted start position accounts for stripped whitespace
        adjusted_start_pos = start_pos + strip_offset
        # Adjusted end position (content length from adjusted start)
        adjusted_end_pos = adjusted_start_pos + len(content)
        
        content_tokens = estimate_tokens(content)
        
        # SUB-CHUNKING: If this section is too large, split it into sub-sections
        MAX_SECTION_TOKENS = 30_000
        
        if content_tokens > MAX_SECTION_TOKENS:
            # This section is too large - split it
            num_sub_chunks = max(2, content_tokens // MAX_SECTION_TOKENS + 1)
            target_chunk_size = len(content) // num_sub_chunks
            
            print(f"[Extractor]   📦 Sub-chunking '{title}' ({content_tokens:,} tokens) into {num_sub_chunks} parts")
            
            # Track actual positions to avoid gaps
            actual_start = 0
            sub_idx = 0
            
            while actual_start < len(content) and sub_idx < num_sub_chunks:
                # Calculate target end position
                if sub_idx < num_sub_chunks - 1:
                    target_end = actual_start + target_chunk_size
                    
                    # Try to break at paragraph boundary (double newline)
                    break_point = content.find('\n\n', target_end - 500, target_end + 500)
                    if break_point != -1:
                        sub_end = break_point
                    else:
                        # Try single newline
                        break_point = content.find('\n', target_end - 200, target_end + 200)
                        if break_point != -1:
                            sub_end = break_point
                        else:
                            sub_end = target_end
                else:
                    # Last chunk - take everything remaining
                    sub_end = len(content)
                
                sub_content = content[actual_start:sub_end].strip()
                
                # Skip empty chunks
                if not sub_content:
                    actual_start = sub_end
                    continue
                    
                sub_title = f"{title} (Part {sub_idx + 1}/{num_sub_chunks})"
                
                # Estimate fields in sub-chunk
                checkbox_count = len(re.findall(r'[☐☑□■\[\]]', sub_content))
                blank_count = len(re.findall(r'_{3,}|\.{5,}', sub_content))
                table_rows = len(re.findall(r'^\|[^-].*\|$', sub_content, re.MULTILINE))
                estimated_fields = checkbox_count + blank_count + int(table_rows * 0.5)
                
                if estimated_fields == 0:
                    estimated_fields = len(sub_content) // 80
                
                # Calculate absolute coordinates using adjusted start (accounts for strip offset)
                abs_start = adjusted_start_pos + actual_start
                abs_end = adjusted_start_pos + sub_end
                
                sections.append(SectionAnchor(
                    title=sub_title,
                    section_number=f"{i+1}.{sub_idx+1}",
                    start_anchor=sub_content[:100] if sub_content else "",
                    end_anchor=sub_content[-100:] if sub_content else "",
                    estimated_density="heavy",
                    has_tables='|' in sub_content and '---' in sub_content,
                    estimated_field_count=estimated_fields,
                    start_index_hint=abs_start,
                    end_index_hint=abs_end
                ))
                
                print(f"[Extractor]     Part {sub_idx + 1}: chars {abs_start:,}-{abs_end:,} ({abs_end - abs_start:,} chars)")
                
                
                actual_start = sub_end
                sub_idx += 1
        else:
            
            start_anchor = content[:100] if content else ""
            end_anchor = content[-100:] if content else ""
            
            # Estimate field density
            content_len = len(content)
            checkbox_count = len(re.findall(r'[☐☑□■\[\]]', content))
            blank_count = len(re.findall(r'_{3,}|\.{5,}', content))
            table_rows = len(re.findall(r'^\|[^-].*\|$', content, re.MULTILINE))
            
            estimated_fields = checkbox_count + blank_count + int(table_rows * 0.5)
            
            if estimated_fields == 0:
                if content_len < 500:
                    estimated_fields = max(0, content_len // 150)
                elif content_len < 3000:
                    estimated_fields = content_len // 100
                else:
                    estimated_fields = content_len // 80
            
            # Determine density
            if estimated_fields < 10:
                density = "light"
            elif estimated_fields < 30:
                density = "medium"
            else:
                density = "heavy"
            
            sections.append(SectionAnchor(
                title=title,
                section_number=None,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
                estimated_density=density,
                has_tables='|' in content and '---' in content,
                estimated_field_count=estimated_fields,
                start_index_hint=adjusted_start_pos,
                end_index_hint=adjusted_end_pos
            ))
    
    print(f"[Extractor]   Regex found {len(sections)} sections")
    
    return DocumentStructure(
        document_title=doc_title or "Untitled Document",
        total_sections=len(sections),
        sections=sections
    )


def extract_structure(markdown: str, config: ExtractionConfig = None, use_regex: bool = True) -> DocumentStructure:
    token_count = estimate_tokens(markdown)
    
    # Use config limits or defaults
    model_context = config.model.context_tokens if config else MODEL_CONTEXT_TOKENS
    max_output = config.model.max_output_tokens if config else MAX_OUTPUT_TOKENS
    
    # Smart Traffic Cop: Determine optimal strategy (kept for logging)
    strategy = DocumentStrategy.determine(
        doc_tokens=token_count,
        model_context=model_context,
        max_output=max_output
    )
    
    print(f"[Extractor]   Strategy: {strategy.mode} (utilization: {strategy.utilization_ratio:.1%})")
    print(f"[Extractor]   Regex-based structure extraction (fast mode)")
    
    # Always use regex - it's faster, deterministic, and handles all cases
    return _extract_structure_regex(markdown, min_header_level=2)


def validate_structure_coverage(
    markdown: str, 
    structure: DocumentStructure
) -> List[str]:
    """Validate structure coverage using GPS coordinates to avoid false positives."""
    warnings = []
    # Use named constant from constants.py
    GAP_THRESHOLD = GAP_WARNING_THRESHOLD_CHARS
    sorted_sections = sorted(structure.sections, key=lambda s: s.start_index_hint or 0)
    
    for i in range(len(sorted_sections) - 1):
        current_section = sorted_sections[i]
        next_section = sorted_sections[i + 1]

        current_search_start = current_section.start_index_hint or 0
        
        
        current_search_start = max(0, current_search_start - 2000)
        
       
        end_pos = markdown.find(current_section.end_anchor, current_search_start)
        
        if end_pos == -1:
            # Fallback: Try full search if local search failed
            end_pos = markdown.find(current_section.end_anchor)
            
        if end_pos == -1:
            warnings.append(
                f"Could not locate end anchor for '{current_section.title}'"
            )
            continue
            
        # Move pos to end of the anchor text
        end_pos += len(current_section.end_anchor)
        
      
        next_search_start = next_section.start_index_hint or end_pos
        next_search_start = max(end_pos, next_search_start - 2000)
        
        start_pos = markdown.find(next_section.start_anchor, next_search_start)
        
        if start_pos == -1:
            # Fallback: Try searching from end of previous section
            start_pos = markdown.find(next_section.start_anchor, end_pos)
            
        if start_pos == -1:
            warnings.append(
                f"Could not locate start anchor for '{next_section.title}'"
            )
            continue
        
     
        gap_size = start_pos - end_pos
        if gap_size > GAP_THRESHOLD:
            gap_content = markdown[end_pos:start_pos].strip()
            # Ignore gaps that are just ### Headers or lines (common artifacts)
            clean_gap = re.sub(r'[#\-\*_=]+', '', gap_content).strip()
            
            if len(clean_gap) > SUBSTANTIVE_GAP_THRESHOLD_CHARS:  
                warnings.append(
                    f"Potential missed content ({gap_size} chars) between "
                    f"'{current_section.title}' and '{next_section.title}'"
                )
    
    return warnings

def group_sections_into_batches(
    structure: DocumentStructure,
    config: ExtractionConfig = DEFAULT_CONFIG
) -> List[List[SectionInfo]]:

    total_sections = len(structure.sections)
    if total_sections > 0:
      
        optimal_size = -(-total_sections // config.concurrent_batch_limit)  # Ceiling division
        target_batch_size = min(optimal_size, config.max_sections_per_batch)
        target_batch_size = max(1, target_batch_size)
        
        print(f"[Extractor]   Load Balancing: {total_sections} sections across {config.concurrent_batch_limit} slots")
        print(f"[Extractor]   Target Batch Size: {target_batch_size} (Max: {config.max_sections_per_batch})")
    else:
        target_batch_size = config.max_sections_per_batch

    batches: List[List[SectionInfo]] = []
    current_batch: List[SectionInfo] = []
    current_estimated_output = 0
    
    for i, section in enumerate(structure.sections):
        # Estimate output for this section
        if section.estimated_density == "heavy" or section.has_tables:
            section_output = section.estimated_field_count * ESTIMATED_TOKENS_PER_FIELD + ESTIMATED_TOKENS_PER_TABLE
        elif section.estimated_density == "medium":
            section_output = section.estimated_field_count * ESTIMATED_TOKENS_PER_FIELD
        else:  # light
            section_output = int(section.estimated_field_count * ESTIMATED_TOKENS_PER_FIELD * 0.7)
        
        section_output += ESTIMATED_TOKENS_PER_SECTION  # Overhead

        would_exceed_tokens = (current_estimated_output + section_output) > config.safe_output_tokens
        would_exceed_count = len(current_batch) >= target_batch_size
        
        # Get next section for boundary detection
        next_section = structure.sections[i + 1] if i + 1 < len(structure.sections) else None
        
        # Create section info from anchor
        section_info = SectionInfo.from_anchor(section, next_section)
        
        if current_batch and (would_exceed_tokens or would_exceed_count):
            # Finalize current batch, start new one
            batches.append(current_batch)
            current_batch = [section_info]
            current_estimated_output = section_output
        else:
            # Add to current batch
            current_batch.append(section_info)
            current_estimated_output += section_output
    
    # Don't forget last batch
    if current_batch:
        batches.append(current_batch)
    
    return batches

def extract_batch(
    markdown: str, 
    section_info: List[SectionInfo],
    batch_id: int,
    config: ExtractionConfig = DEFAULT_CONFIG,
    continuity_hint: Optional['BatchContinuityHint'] = None
) -> Tuple[BatchResult, Optional['BatchContinuityHint']]:
    """Extract form data from a batch of sections."""
    from prompts import build_targeted_extraction_prompt
    
    client = get_client()
    section_titles = [info.title for info in section_info]
    prompt = build_targeted_extraction_prompt(section_titles, continuity_hint=continuity_hint)
    
    # Inject context note into prompt if continuing from previous batch
    if continuity_hint and continuity_hint.context_note:
        prompt += f"\n\n## CRITICAL CONTEXT FROM PREVIOUS BATCH\n{continuity_hint.context_note}\n- Treat the first table rows as a CONTINUATION of the previous table.\n- Do NOT create a new table header.\n- Do NOT restart numbering."
    
    section_contents = []
    
    if continuity_hint and continuity_hint.prepend_content:
        # Add markdown separation to ensure proper parsing
        section_contents.append(f"\n[CONTINUATION FROM PREVIOUS BATCH]\n{continuity_hint.prepend_content}\n")
        print(f"[Extractor]   📎 Prepending continuation content ({len(continuity_hint.prepend_content)} chars)")
    
    for info in section_info:

        if info.start_index_hint is not None and info.end_index_hint is not None:
            # STRICT INDEXING: Direct character slice - no ambiguity
            start = max(0, info.start_index_hint)
            end = min(len(markdown), info.end_index_hint)
            
            # Diagnostic: Show position info
            slice_chars = end - start
            print(f"[Extractor]   📍 '{info.title}': chars {start:,}-{end:,} ({slice_chars:,} chars)")
            
            raw_content = markdown[start:end].strip()
            
            if raw_content:
                # Check if we need to extend for table boundary
                extended_end, _, _ = _extend_to_table_boundary(markdown, end, max_extension=5000)
                if extended_end > end:
                    content = markdown[start:extended_end].strip()
                    print(f"[Extractor]     ↳ Extended to {extended_end:,} for table boundary (+{extended_end - end:,} chars)")
                else:
                    content = raw_content
                
                section_contents.append(content)
                print(f"[Extractor]   ✓ Strict slice: '{info.title}' ({len(content):,} chars)")
            else:
                print(f"[Extractor]   ⚠️  Warning: Strict slice for '{info.title}' was empty, falling back to anchors")
                # Fallback to anchor search
                content = _extract_section_content_by_anchors(
                    markdown, info.start_anchor, info.end_anchor, info.start_index_hint, info.title, info.next_section_title
                )
                if content:
                    section_contents.append(content)
                else:
                    section_contents.append(f"## {info.title}\n[Section empty after slice]")
        else:
            # Fallback: Use anchor-based search (legacy path for non-sub-chunked sections)
            content = _extract_section_content_by_anchors(
                markdown, info.start_anchor, info.end_anchor, info.start_index_hint, info.title, info.next_section_title
            )
            if content:
                section_contents.append(content)
            else:
                # Fallback: include a warning
                print(f"[Extractor]   ⚠️  Warning: Could not locate section '{info.title}' using anchors")
                section_contents.append(f"## {info.title}\n[Section not found - anchor mismatch]")
    
    focused_content = "\n\n".join(section_contents)
    
    if batch_id == 0:
        
        preamble = markdown[:2000]
        focused_content = preamble + "\n\n[...document continues...]\n\n" + focused_content
    
    
    debug_dir = Path("debug_markdown")
    debug_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_md_path = debug_dir / f"batch_{batch_id:03d}_{timestamp}.md"
    batch_md_path.write_text(focused_content, encoding="utf-8")
    
    completion = client.beta.chat.completions.parse(
        model=config.model.name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": focused_content},
            {"role": "user", "content": "CRITICAL: Extract EVERYTHING from these sections - all tables, all fields, all content. Do not skip anything."}
        ],
        response_format=ChunkExtraction,
        reasoning_effort="low",  
        max_completion_tokens=config.model.max_output_tokens
    )
    
    parsed = completion.choices[0].message.parsed
    finish_reason = completion.choices[0].finish_reason
    output_tokens = completion.usage.completion_tokens
    input_tokens = completion.usage.prompt_tokens
    
    # Log token savings
    full_doc_tokens = estimate_tokens(markdown)
    actual_tokens = input_tokens
    saved_tokens = full_doc_tokens - actual_tokens
    
    print(f"[Extractor]   Batch {batch_id}: Sent {actual_tokens:,} tokens (saved {saved_tokens:,} vs full doc)")
    
    next_hint = None
    
    if section_info:  # If we have sections
        last_section_info = section_info[-1]

        last_section_end_pos = last_section_info.start_index_hint or 0
       
        last_content_end = markdown.find(last_section_info.end_anchor, last_section_end_pos)
        
        if last_content_end != -1:
            # Check if extending would find a split table
            extended_end, table_header, table_name = _extend_to_table_boundary(
                markdown, 
                last_content_end,
                max_extension=20000  # Conservative limit
            )
            
            
            if table_header and table_name:
                next_hint = BatchContinuityHint(
                    prepend_content=table_header,
                    open_table_name=table_name,
                    context_note=f"Continuing table '{table_name}' from previous batch",
                    last_element_type="table"
                )
                print(f"[Extractor]    Table '{table_name}' continues to next batch")
    
    batch_result = BatchResult(
        batch_id=batch_id,
        sections_requested=section_titles,
        sections_extracted=parsed.sections if parsed else [],
        finish_reason=finish_reason,
        output_tokens_used=output_tokens
    )
    
    return batch_result, next_hint


def extract_batch_with_retry(
    markdown: str, 
    section_info: List[SectionInfo],
    batch_id: int,
    max_retries: int = 3,
    continuity_hint: Optional[BatchContinuityHint] = None
) -> Tuple[List[FormSection], Optional[BatchContinuityHint]]:
   
    for attempt in range(max_retries + 1):
        try:
            result, next_hint = extract_batch(markdown, section_info, batch_id, continuity_hint=continuity_hint)
            
            if result.finish_reason == 'stop':
                return result.sections_extracted, next_hint
            
            if result.finish_reason == 'length':
                print(f"[Extractor]   Batch {batch_id} truncated, splitting...")
                
                if len(section_info) == 1:
                    # Can't split further - single section too large
                    print(f"[Extractor]  Section '{section_info[0].title}' too large for single extraction")
                    return result.sections_extracted, next_hint  # Return partial with hint
                
                # Split in half and retry
                mid = len(section_info) // 2
                first_half = section_info[:mid]
                second_half = section_info[mid:]
                
                results_first, hint_first = extract_batch_with_retry(
                    markdown, first_half, batch_id * 10 + 1, max_retries, continuity_hint
                )
                
                results_second, hint_second = extract_batch_with_retry(
                    markdown, second_half, batch_id * 10 + 2, max_retries, hint_first
                )
                
                return results_first + results_second, hint_second
            
           
            print(f"[Extractor]   Batch {batch_id} unknown finish: {result.finish_reason}")
            return result.sections_extracted, next_hint
            
        except Exception as e:
            error_str = str(e)
            # Rate limit handling
            if "429" in error_str or "rate_limit" in error_str.lower():
                delay = RATE_LIMIT_RETRY_BASE_DELAY * (2 ** attempt)
                print(f"[Extractor]   Batch {batch_id} rate limit hit, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})...")
                time.sleep(delay)
                continue
            
            # Other errors
            if attempt < max_retries:
                print(f"[Extractor]   Batch {batch_id} failed (attempt {attempt + 1}), retrying...")
                time.sleep(1)
                continue
            
            print(f"[Extractor]  Batch {batch_id} failed after {max_retries + 1} attempts: {e}")
            return [], None  # Return empty on failure
    
    return [], None


def merge_batch_results(
    structure: DocumentStructure,
    batch_results: List[List[FormSection]]
) -> List[FormSection]:
    
    all_sections = []
    section_counter = 0
    
    for batch in batch_results:
        for section in batch:
            section_counter += 1
            
            # Assign section ID
            section.id = f"section_{section_counter}"
            
            # Assign field IDs within section
            field_counter = 0
            table_counter = 0
            
            for item in section.content:
                if hasattr(item, 'id'):
                    if hasattr(item, 'type') and item.type == 'table':
                        table_counter += 1
                        item.id = f"s{section_counter}_table_{table_counter}"
                    else:
                        field_counter += 1
                        item.id = f"s{section_counter}_field_{field_counter}"
            
            all_sections.append(section)
    
    return all_sections

def create_streaming_chunks(
    markdown: str,
    max_tokens: int = None,
    overlap_chars: int = 500
) -> List[Tuple[str, int, int]]:

    from schemas import StreamingState
    
    if max_tokens is None:
        max_tokens = CHUNK_MAX_SIZE_TOKENS
    
    # Convert tokens to chars (conservative: 2.5 chars/token)
    max_chars = int(max_tokens * 2.5)
    
    total_len = len(markdown)
    chunks = []
    current_pos = 0
    
    while current_pos < total_len:
        # Calculate tentative end position
        tentative_end = min(current_pos + max_chars, total_len)
        
        if tentative_end >= total_len:
            # Last chunk - take everything remaining
            chunks.append((markdown[current_pos:], current_pos, total_len))
            break
        
        # Find a safe cut point (extend to table boundary, then find paragraph break)
        safe_end, table_header, table_name = _extend_to_table_boundary(
            markdown, tentative_end, max_extension=20000
        )
        
        # If table was extended significantly, use that boundary
        if safe_end > tentative_end + 100:
            actual_end = safe_end
        else:
            # Find a clean paragraph/section break
            actual_end = _find_safe_break_point(markdown, tentative_end, current_pos)
        
        # Extract chunk
        chunk_text = markdown[current_pos:actual_end]
        chunks.append((chunk_text, current_pos, actual_end))

        current_pos = actual_end
    
    return chunks


def _find_safe_break_point(markdown: str, target_pos: int, min_pos: int) -> int:
    """Find a safe place to break (paragraph, section header, or sentence)."""
    search_start = max(min_pos, target_pos - 2000)
    search_region = markdown[search_start:target_pos]
    
    # Priority 1: Section header (## or #)
    header_match = None
    for match in re.finditer(r'\n(#{1,3}\s+[^\n]+)\n', search_region):
        header_match = match
    if header_match:
        return search_start + header_match.start() + 1
    
    # Priority 2: Double newline (paragraph break)
    last_para = search_region.rfind('\n\n')
    if last_para != -1 and last_para > len(search_region) // 2:
        return search_start + last_para + 2
    
    # Priority 3: Single newline
    last_newline = search_region.rfind('\n')
    if last_newline != -1:
        return search_start + last_newline + 1
    
    # Fallback: use target position
    return target_pos


def build_streaming_prompt(state: 'StreamingState') -> str:
    """Build a prompt with state injection for streaming extraction."""
    from prompts import (
        VERBATIM_EXTRACTION_RULES, FIELD_TYPE_RULES, TABLE_HANDLING_RULES,
        FIELD_PROPERTIES_RULES, PLACEHOLDER_RULES, CONDITIONAL_LOGIC_RULES,
        FIELD_GROUPING_RULES, SECTION_AUDIENCE_RULES, ID_RULES
    )
    
    context_block = ""
    
    if state.chunk_index > 0:
        # This is a continuation chunk - inject context
        context_parts = ["\n##  CONTINUATION CONTEXT (CRITICAL)"]
        context_parts.append(f"This is chunk {state.chunk_index + 1} of a large document.")
        
        if state.active_section_title:
            context_parts.append(f"""
**ACTIVE SECTION:** "{state.active_section_title}"
The previous chunk ended inside this section. Content at the start of this chunk
belongs to "{state.active_section_title}" - do NOT create a new section for it.
Continue extracting into the same section title.""")
        
        if state.table_in_progress and state.table_name:
            context_parts.append(f"""
**TABLE IN PROGRESS:** "{state.table_name}"
A table was split at the chunk boundary. The rows at the start of this chunk
are ADDITIONAL ROWS for "{state.table_name}". Do NOT create a new table.""")
            if state.table_headers:
                headers_str = " | ".join(state.table_headers)
                context_parts.append(f"Table columns: {headers_str}")
        
        if state.overlap_text:
            context_parts.append(f"""
**OVERLAP (for context only, already extracted):**
```
{state.overlap_text[:200]}...
```
Start extracting AFTER this overlap text.""")
        
        context_block = "\n".join(context_parts) + "\n"
    
    return f"""You are extracting form fields from a document chunk.
{context_block}
## EXTRACTION RULES

{VERBATIM_EXTRACTION_RULES}

{FIELD_TYPE_RULES}

{TABLE_HANDLING_RULES}

{FIELD_PROPERTIES_RULES}

{PLACEHOLDER_RULES}

{CONDITIONAL_LOGIC_RULES}

{FIELD_GROUPING_RULES}

{SECTION_AUDIENCE_RULES}

{ID_RULES}

## OUTPUT
Return a ChunkExtraction object with ALL sections found in this chunk.

## CRITICAL
- Extract EVERY field - missing fields = data loss
- If content has no section header, use the ACTIVE SECTION from context
- When in doubt, include it
- Verbatim extraction - do not summarize"""


def extract_streaming_chunk(
    chunk_text: str,
    chunk_index: int,
    state: 'StreamingState',
    config: ExtractionConfig = DEFAULT_CONFIG
) -> Tuple[List[FormSection], 'StreamingState']:
    """
    Extract a single chunk using streaming approach with state injection.
    
    Returns: (sections, updated_state)
    """
    from schemas import StreamingState, ChunkExtraction
    
    client = get_client()
    prompt = build_streaming_prompt(state)
    
    # Save debug output
    debug_dir = Path("debug_markdown")
    debug_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    debug_path = debug_dir / f"stream_{chunk_index:03d}_{timestamp}.md"
    debug_path.write_text(chunk_text, encoding="utf-8")
    
    completion = client.beta.chat.completions.parse(
        model=config.model.name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk_text},
            {"role": "user", "content": "Extract ALL sections and fields from this chunk. If continuing a section from context, use the SAME section title."}
        ],
        response_format=ChunkExtraction,
        reasoning_effort="low",
        max_completion_tokens=min(config.model.max_output_tokens, 128_000)  # API limit is 128k
    )
    
    finish_reason = completion.choices[0].finish_reason
    parsed = completion.choices[0].message.parsed
    
    if parsed is None:
        print(f"[Extractor]     Chunk {chunk_index} returned null, skipping")
        return [], state
    
    sections = parsed.sections if parsed.sections else []
    
    # Update state for next chunk
    new_state = StreamingState(
        chunk_index=chunk_index + 1,
        total_sections_extracted=state.total_sections_extracted + len(sections)
    )
    
    if sections:
        last_section = sections[-1]
        new_state.active_section_title = last_section.title
        new_state.active_section_id = last_section.id
        
        # Check if last item is a table (might be split)
        if last_section.content:
            last_item = last_section.content[-1]
            if hasattr(last_item, 'type') and last_item.type == 'table':
                # Check if chunk ends mid-table by looking at the raw text
                if chunk_text.rstrip().endswith('|'):
                    new_state.table_in_progress = True
                    new_state.table_name = getattr(last_item, 'title', None) or last_section.title
                    new_state.table_headers = getattr(last_item, 'headers', None)
    
    # Capture overlap for next chunk
    new_state.overlap_text = chunk_text[-500:] if len(chunk_text) > 500 else chunk_text
    
    if finish_reason == 'length':
        print(f"[Extractor]     Chunk {chunk_index} was truncated")
    
    return sections, new_state


def merge_streaming_sections(all_sections: List[FormSection]) -> List[FormSection]:
    """
    Merge sections with the same title that were split across chunks.
    
    This handles the "Severed Head" problem where a section continues
    across chunk boundaries.
    """
    if not all_sections:
        return []
    
    merged = []
    section_map: Dict[str, FormSection] = {}
    
    for section in all_sections:
        title_key = section.title.strip().lower()
        
        # Check for continuation markers
        is_continuation = any(marker in title_key for marker in [
            '(continued)', '(cont)', '- continued', 'continuation'
        ])
        
        if is_continuation:
            # Clean title and find original
            base_title = re.sub(r'\s*\(continued?\)|\s*-\s*continued?|\s*continuation', '', 
                               section.title, flags=re.IGNORECASE).strip()
            base_key = base_title.lower()
            
            if base_key in section_map:
                # Append content to existing section
                section_map[base_key].content.extend(section.content)
                print(f"[Merger]   Merged continuation into '{base_title}'")
                continue
        
        if title_key in section_map:
            # Same title seen before - merge content
            section_map[title_key].content.extend(section.content)
            print(f"[Merger]   Merged duplicate section '{section.title}'")
        else:
            # New section
            section_map[title_key] = section
            merged.append(section)
    
    # Re-number sections and fields
    for i, section in enumerate(merged, 1):
        section.id = f"section_{i}"
        field_counter = 0
        table_counter = 0
        
        for item in section.content:
            if hasattr(item, 'id'):
                if hasattr(item, 'type') and item.type == 'table':
                    table_counter += 1
                    item.id = f"s{i}_table_{table_counter}"
                else:
                    field_counter += 1
                    item.id = f"s{i}_field_{field_counter}"
    
    return merged


def execute_streaming_extraction(
    markdown: str,
    config: ExtractionConfig = DEFAULT_CONFIG
) -> List[FormSection]:

    from schemas import StreamingState
    
    # Create physical chunks
    chunks = create_streaming_chunks(markdown, max_tokens=config.model.context_tokens // 3)
    print(f"[Extractor] Streaming mode: {len(chunks)} chunks")
    
    all_sections = []
    state = StreamingState()
    
    for i, (chunk_text, start_pos, end_pos) in enumerate(chunks):
        chunk_tokens = estimate_tokens(chunk_text)
        print(f"[Extractor]   Chunk {i+1}/{len(chunks)}: chars {start_pos:,}-{end_pos:,} ({chunk_tokens:,} tokens)")
        
        sections, state = extract_streaming_chunk(chunk_text, i, state, config)
        all_sections.extend(sections)
        
        print(f"[Extractor]     → {len(sections)} sections extracted")
        
        # Rate limit courtesy
        if i < len(chunks) - 1:
            time.sleep(0.5)
    
    # Merge split sections
    merged_sections = merge_streaming_sections(all_sections)
    print(f"[Extractor]   Merged: {len(all_sections)} → {len(merged_sections)} sections")
    
    return merged_sections

def get_dynamic_validation_threshold(markdown: str) -> float:

    # Count form field indicators
    underscores = markdown.count("___")
    checkboxes = markdown.count("[ ]") + markdown.count("☐") + markdown.count("☑")
    dots = len(re.findall(r'\.{5,}', markdown))  
    table_pipes = markdown.count("|") // 4  
    
    # Total form signals
    total_signals = underscores + checkboxes + dots + table_pipes
    
    # Rough token estimate (4 chars ≈ 1 token)
    estimated_tokens = len(markdown) / 4
    
    # Calculate signal density (signals per 100 tokens)
    if estimated_tokens == 0:
        density = 0
    else:
        density = (total_signals / estimated_tokens) * 100
    

    if density > 5.0:
        threshold = 0.10
        print(f"[Validator]   Form density: HIGH ({density:.2f} signals/100 tokens) - threshold: {threshold:.1%}")
    elif density > 1.0:
     
        threshold = 0.05
        print(f"[Validator]   Form density: MEDIUM ({density:.2f} signals/100 tokens) - threshold: {threshold:.1%}")
    else:
        threshold = 0.005
        print(f"[Validator]   Form density: LOW ({density:.2f} signals/100 tokens) - threshold: {threshold:.1%}")
    
    return threshold


def validate_compression_ratio(
    input_markdown: str,
    output_json: str,
    threshold: Optional[float] = None
) -> dict:

    input_tokens = estimate_tokens(input_markdown)
    output_tokens = estimate_tokens(output_json)
    
    ratio = output_tokens / input_tokens if input_tokens > 0 else 0
    
    
    if threshold is None:
        threshold = get_dynamic_validation_threshold(input_markdown)
    
    is_valid = ratio >= threshold
    
    warnings = []
    if not is_valid:
        warnings.append(
            f"Suspicious compression ratio: {ratio:.1%} (output/input) - threshold: {threshold:.1%}. "
            f"This may indicate missing data, hallucination, or extraction failure."
        )
    elif ratio > 2.0:
        
        warnings.append(
            f"High expansion ratio: {ratio:.1%} (output/input). Check for duplicated content."
        )
    
    return {
        "is_valid": is_valid,
        "ratio": ratio,
        "threshold": threshold,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "warnings": warnings
    }


def validate_extraction_completeness(
    markdown: str, 
    sections: List[FormSection]
) -> dict:

    # Count extracted fields
    extracted_count = 0
    for section in sections:
        for item in section.content:
            if hasattr(item, 'type'):
                if item.type == 'table':
                    
                    if hasattr(item, 'rows'):
                        extracted_count += len(item.rows)
                else:
                    extracted_count += 1
    
    # Estimate expected fields from markdown
    checkbox_count = len(re.findall(r'[☐☑□■]', markdown))
    bracket_count = len(re.findall(r'\[\s*[xX]?\s*\]', markdown))
    blank_count = len(re.findall(r'_{3,}|\.{3,}', markdown))
    table_rows = len(re.findall(r'^\|[^-].*\|$', markdown, re.MULTILINE))
    
    expected_count = checkbox_count + bracket_count + blank_count + int(table_rows * 0.5)
    
    # Calculate coverage
    coverage = (extracted_count / expected_count * 100) if expected_count > 0 else 100
    is_complete = coverage >= EXTRACTION_COMPLETE_THRESHOLD_PCT
    
    # Generate warnings
    warnings = []
    if not is_complete:
        warnings.append(
            f"Low coverage: {extracted_count} extracted vs {expected_count} expected ({coverage:.1f}%)"
        )
    
    if extracted_count > expected_count * DUPLICATE_WARNING_THRESHOLD_RATIO:
        warnings.append(
            f"Unusually high extraction: {extracted_count} vs {expected_count} expected - possible duplicates"
        )
    
    return {
        "extracted_count": extracted_count,
        "expected_count": expected_count,
        "coverage_percentage": coverage,
        "is_complete": is_complete,
        "warnings": warnings
    }


def _extract_metadata_regex(text: str) -> dict:
    patterns = {
        'version': r'[Vv]ersion\s*:?\s*([\d.]+)',
        'cricos_code': r'CRICOS\s*[Cc]ode\s*:?\s*(\w+)',
        'provider_code': r'[Pp]rovider\s*[Cc]ode\s*:?\s*(\d+)',
        'abn': r'ABN\s*:?\s*([\d\s]+)',
        'rto_code': r'RTO\s*(?:[Cc]ode\s*)?:?\s*(\d+)',
        'organization': r'(?:RTO|Organization|Organisation)\s*:?\s*([A-Z][A-Za-z\s&]+(?:Ltd|Pty|Institute|College|Training|Academy)?)',
    }
    
    metadata = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            # Clean up ABN (remove spaces)
            if key == 'abn':
                value = re.sub(r'\s+', '', value)
            metadata[key] = value
    
    return metadata

def _extract_full_document(markdown: str, config: ExtractionConfig = DEFAULT_CONFIG) -> FormDocument:
    """Extract complete FormDocument (used for first chunk or small docs)."""
    client = get_client()
    
    # Pre-extract metadata with regex (deterministic)
    regex_metadata = _extract_metadata_regex(markdown)
    if regex_metadata:
        print(f"[Extractor] Regex-extracted metadata: {list(regex_metadata.keys())}")
    
    max_retries = 2  # Retry up to 2 times for low coverage
    last_result = None
    
    for attempt in range(max_retries + 1):
        completion = client.beta.chat.completions.parse(
            model=config.model.name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": markdown},
                # Reinforce complete extraction
                {"role": "user", "content": "CRITICAL: Extract EVERYTHING. All sections, all tables, all fields, all content from start to end. Do not skip or summarize anything. Complete extraction required."}
            ],
            response_format=FormDocument,
            reasoning_effort="low",  # Literal extraction, no overthinking
            max_completion_tokens=config.model.max_output_tokens
        )
        
        # Check if response was truncated
        finish_reason = completion.choices[0].finish_reason
        usage = completion.usage
        
        if finish_reason == 'length':
            # Extract token usage details for better error message
            completion_tokens = usage.completion_tokens if usage else 0
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens if usage and usage.completion_tokens_details else 0
            
            error_msg = f"Could not parse response content as the length limit was reached - {usage}"
            print(f"[Extractor]   Single-pass extraction truncated!")
            print(f"[Extractor]   Completion tokens: {completion_tokens:,} (reasoning: {reasoning_tokens:,})")
            print(f"[Extractor]   Falling back to targeted extraction...")
            raise ValueError(error_msg)
        
        if finish_reason != 'stop':
            print(f"[Extractor]   Unusual finish_reason: {finish_reason}")
        
        parsed_data = completion.choices[0].message.parsed
        
        if parsed_data is None:
            refusal = completion.choices[0].message.refusal
            if refusal:
                raise ValueError(f"Model refused: {refusal}")
            raise ValueError("Extraction failed")
        
        # Quick coverage check - count sections and fields
        section_count = len(parsed_data.sections) if parsed_data.sections else 0
        field_count = sum(len(s.content) for s in parsed_data.sections) if parsed_data.sections else 0
        
        # If we got reasonable output, accept it
        if section_count >= 1 and field_count >= 3:
            last_result = parsed_data
            break
        
        # Low coverage - retry if we have attempts left
        if attempt < max_retries:
            print(f"[Extractor]     Low extraction ({section_count} sections, {field_count} fields), retrying...")
            last_result = parsed_data
            continue
        else:
            # Use whatever we got on last attempt
            last_result = parsed_data
    
    # Override LLM metadata with regex-extracted values (more reliable)
    for key, value in regex_metadata.items():
        if hasattr(last_result.metadata, key):
            old_value = getattr(last_result.metadata, key)
            setattr(last_result.metadata, key, value)
            if old_value and old_value != value:
                print(f"[Extractor]   Corrected {key}: '{old_value}' → '{value}'")
    
    return last_result


def extract_form_data(markdown_content: str, config: ExtractionConfig = None) -> FormDocument:

    from pipeline import ExtractionPipeline
    from config import get_config_for_document, DEFAULT_CONFIG
    
    # Use hybrid mode if no config provided
    if config is None:
        doc_tokens = estimate_tokens(markdown_content)
        config = get_config_for_document(doc_tokens)
    
    pipeline = ExtractionPipeline(markdown_content, config)
    result = pipeline.run()
    
    return result.document

def extract_document_data(markdown_content: str) -> FormDocument:
    """Legacy entry point - uses hybrid configuration."""
    return extract_form_data(markdown_content)
