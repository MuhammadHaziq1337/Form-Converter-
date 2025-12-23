"""Composable prompt components for consistent extraction instructions."""

from typing import List, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from schemas import ChunkContext, BatchContinuityHint


VERBATIM_EXTRACTION_RULES = """## CORE PRINCIPLE - VERBATIM EXTRACTION
Extract text EXACTLY as written in the document. Do NOT summarize, paraphrase, or shorten.
- Copy the EXACT words, bullet points, numbered lists
- Do NOT correct typos (e.g., if "Emial:", extract "Emial:")
- Do NOT expand abbreviations (e.g., "N/A" stays "N/A")
- Do NOT infer values for blank checkboxes
- Use \\n for line breaks within text

## CRITICAL: INPUT FIELD DETECTION
Look for visual patterns that indicate input fields:
- Dotted lines: `...........` or `.................` → **Form field** (UNLESS followed by a page number like "Page 5")
- Underscores: `____` or `________` → **ALWAYS a form field**
- Bracketed blanks: `[____]` or `[________]` → **Form field**
- Pattern: `Label: ..........` → Extract as type="text", label="Label", editable=true
- Pattern: `Signature: ..........` → type="signature"
- Pattern: `Date: ..........` → type="date"
- Pattern: `Student ID /Date of Birth..........` → Split into TWO fields

**EXCLUSIONS - NOT form fields:**
- Table of Contents leaders: `Chapter 1 ........................ Page 5` → This is a label, NOT a field
- Index entries: `Term .................. 123` → Label only
- If dots connect two pieces of static text with a page number, it's NOT a field

**Example transformations:**
```
Candidate name: ..................... Student ID /Date of Birth..................
```
Becomes TWO fields:
1. type="text", label="Candidate name", editable=true, placeholder="Enter candidate name"
2. type="text", label="Student ID /Date of Birth", editable=true, placeholder="Enter student ID or date of birth"

```
Signature: ...................
```
Becomes:
type="signature", label="Signature", editable=true

```
DATE: ...................
```
Becomes:
type="date", label="DATE", editable=true"""


FIELD_TYPE_RULES = """### Field Type Detection

| Type | Use When |
|------|----------|
| label | Instructions, static text (editable: false) |
| text | Single-line input (name, short answer) |
| textarea | Multi-line input (comments, descriptions) |
| checkbox | Single tick box ("I agree to terms") |
| checkbox_group | Multiple selections ("select all that apply") |
| radio | Mutually exclusive choice (Yes/No, gender) |
| select | Dropdown menu |
| date | Date input |
| number | Numeric input |
| signature | Signature field |

### Checkbox vs Radio Detection
- [ ] Yes [ ] No → type: "radio", options: ["Yes", "No"]
- ☐ Yes ☐ No → type: "radio", options: ["Yes", "No"]
- ⬜ Yes ⬜ No → type: "radio", options: ["Yes", "No"]
- "Select all that apply" → type: "checkbox_group"
- Single standalone [ ] I agree → type: "checkbox"
- Single standalone ⬜ I agree → type: "checkbox" """


TABLE_HANDLING_RULES = """### Table Handling

**EXTRACT ALL TABLES - NO EXCEPTIONS**

Every table in the document must be extracted as type="table".

**Rules:**
- Extract ALL tables exactly as they appear
- Each `TableCell` MUST include the explicit `column` header name
- For checkbox columns (☐, ⬜ or [ ]): set input_type="checkbox", editable=true
- For empty cells: set input_type="text", editable=true  
- For pre-filled cells: set editable=false, include the value
- Preserve row order exactly

**Example:**
```
| Name | Signature | Date |
|------|-----------|------|
|      |           |      |
```
Extract as:
```json
{
  "type": "table",
  "headers": ["Name", "Signature", "Date"],
  "rows": [{"cells": [
    {"column": "Name", "input_type": "text", "editable": true},
    {"column": "Signature", "input_type": "text", "editable": true},
    {"column": "Date", "input_type": "text", "editable": true}
  ]}]
}
```

**NO table should be skipped. Extract everything.**"""


FIELD_PROPERTIES_RULES = """### Field Properties

| Condition | Properties |
|-----------|------------|
| Blank/empty field | editable: true, value: null |
| Pre-filled field | editable: false, value: "the content" |
| Input field | required: true |
| Label/instruction | required: false |"""


PLACEHOLDER_RULES = """### Placeholders
- When you see ______, ......, [____], blank lines
- Generate SEMANTIC placeholder, not raw characters
- "Date of Birth: ____" → placeholder: "Enter date of birth"
- "Name: ............" → placeholder: "Enter full name" """


ID_RULES = """### IDs
- Do NOT generate ID fields
- Leave all IDs as null
- IDs will be auto-generated in post-processing"""


POST_TABLE_RULES = """### Post-Table Content
ALWAYS scan after tables for signatures, dates, and office-use sections:
- "Authorised Person": `text`
- "Signature": `signature`
- "Date": `date`"""

CONDITIONAL_LOGIC_RULES = """### Conditional Fields
When you see patterns indicating field dependencies:
- "If Yes, please specify: ____"
- "If other, please state: ____"  
- "Required if under 18"
- "Complete only if applicable"

Extract the dependent field with:
- `depends_on`: Use "PREV" to reference the immediately preceding field, or "PREV_2", "PREV_3" etc.
- `show_when`: Value that triggers visibility (e.g., "Yes", "Other", "not_empty")

Example:
  Q5. Do you have a disability? [ ] Yes [ ] No
      If yes, please specify: ____

Extract as:
- Field 1: type="radio", label="Do you have a disability?", options=["Yes", "No"]
- Field 2: type="text", label="If yes, please specify:", depends_on="PREV", show_when="Yes"

The system will resolve "PREV" references to actual IDs in post-processing."""


FIELD_GROUPING_RULES = """### Field Grouping
When fields belong together visually (address blocks, name blocks):
- Assign the same `group` string to all related fields
- First field in group sets `group_layout`: "horizontal" or "vertical"

Example address block:
  Street: ____  Suburb: ____  State: ____  Postcode: ____

Extract as:
- Street → group="address", group_layout="horizontal"
- Suburb → group="address"
- State → group="address"  
- Postcode → group="address"

Common groupings: address, name, contact, date_range"""


SECTION_AUDIENCE_RULES = """### Section Audience (CRITICAL)
Identify who fills each section by looking for keywords in section titles and content:

**Assessor/Staff sections:**
- Title contains: "Assessor", "Staff", "Office Use", "Internal", "Training goals"
- Examples: "PART A (Assessor Instructions and Training goals)"
→ Set `filled_by`: "assessor", `instructions`: Extract the assessor instructions

**Applicant/Student/Candidate sections:**
- Title contains: "Candidate", "Student", "Applicant", "Learner", "Guide"
- Examples: "SECTION 2: LLN CANDIDATE GUIDE", "Personal details"
→ Set `filled_by`: "user"

**Third Party sections:**
- Title contains: "Employer", "Referee", "Supervisor", "Witness"
→ Set `filled_by`: "thirdparty"

**Introduction/Information sections:**
- Title contains: "Introduction", "Purpose", "Overview"
- Contains only instructions/explanations, no input fields
→ Leave `filled_by` as null, these are informational

If section has mixed audiences, set to primary audience.
Extract section-specific instructions into the `instructions` field."""


VALIDATION_RULES = """### Validation Rules
When you see explicit validation requirements:

- "Maximum 500 characters" → validation: {max_length: 500}
- "10-digit mobile" → validation: {pattern: "^[0-9]{10}$"}
- "Must be 18 or older" → validation: {max_date: "today-18years"}
- "Between 1 and 100" → validation: {min_value: 1, max_value: 100}

Include custom_error if an error message is specified."""

def build_system_prompt() -> str:
    """Build the main system prompt from components."""
    return f"""You are a high-precision document extraction engine.
Your goal is to convert the provided Markdown document into a structured JSON form, preserving the exact content and structure.

{VERBATIM_EXTRACTION_RULES}

## EXTRACTION RULES

### Metadata & Document Name
- Extract from header/footer text
- Use the FULL organization name
- `filled_by`: "user" (applicant), "assessor" (staff), "thirdparty", or "mixed"

### Sections - COMPLETE EXTRACTION

**EXTRACT EVERY SECTION FROM START TO END**

- Process the ENTIRE document - do not stop early
- Every heading (##, ###) = new section  
- If you see Section 1, 2, 3... you MUST extract ALL of them
- If you see Part A, B, C... you MUST extract ALL of them
- Include ALL content: text, tables, fields, instructions
- Leave `id` fields null (auto-generated later)

**Missing any section = FAILURE**

{SECTION_AUDIENCE_RULES}

### Fields - CRITICAL ANTI-HALLUCINATION RULES

**STOP OVER-EXTRACTING!**

If a paragraph is pure instruction/explanation text with NO input pattern → Extract as ONE label field
If a paragraph has input patterns (dots, underscores) → Extract those as separate fields

**What counts as an input pattern:**
- Dotted lines: `.........` (unless TOC page numbers)
- Underscores: `____`
- Empty table cells with editable: true
- Checkboxes: `[ ]` or `☐`

**What is NOT an input field:**
- Bullet points listing instructions
- Numbered steps explaining a process
- Paragraphs describing requirements
- Reference information (like mapping tables)

**Examples:**

 WRONG: Extract each bullet point as a separate field
```
This assessment must be completed independently:
- You may ask for clarification
- Use clear handwriting
- No time limit
```
DO NOT extract 3 fields here!

✓ RIGHT: Extract as ONE label
type="label", label="This assessment must be completed independently:\n- You may ask for clarification\n- Use clear handwriting\n- No time limit", editable=false

 WRONG: Extract entire paragraph as one field when it has input patterns
```
Candidate name: ..................... Student ID: .....................
```

✓ RIGHT: Split into TWO fields
1. type="text", label="Candidate name", editable=true
2. type="text", label="Student ID", editable=true

Example transformation:
```
Candidate name: ..................... Student ID /Date of Birth..................
Signature: ...................
DATE: ...................
```
Becomes FOUR fields:
1. type="text", label="Candidate name", editable=true
2. type="text", label="Student ID /Date of Birth", editable=true  
3. type="signature", label="Signature", editable=true
4. type="date", label="DATE", editable=true

- Copy field labels EXACTLY
- Use `\\n` for line breaks within labels

{TABLE_HANDLING_RULES}

{POST_TABLE_RULES}

{FIELD_TYPE_RULES}

{FIELD_PROPERTIES_RULES}

{PLACEHOLDER_RULES}

{CONDITIONAL_LOGIC_RULES}

{FIELD_GROUPING_RULES}

{VALIDATION_RULES}

{ID_RULES}

## CRITICAL
You are using Structured Outputs. Populate the schema strictly. Do not output markdown code blocks, just the JSON response."""


def build_targeted_extraction_prompt(
    section_titles: List[str], 
    continuity_hint: 'BatchContinuityHint' = None
) -> str:
    """Build prompt for extracting specific sections."""
    sections_list = "\n".join(f"  • {title}" for title in section_titles)
    
    # Build continuation context if provided
    continuation_block = ""
    if continuity_hint and continuity_hint.open_table_name:
        continuation_block = f"""
##  CRITICAL CONTEXT: TABLE CONTINUATION

You are CONTINUING the table "{continuity_hint.open_table_name}" from the previous batch.

**RULES:**
1. The content at the start belongs to the table "{continuity_hint.open_table_name}"
2. These are ADDITIONAL ROWS for that table - do NOT create a new table
3. Extract these rows as part of the existing "{continuity_hint.open_table_name}" table structure
4. If you see table headers repeated, they are for YOUR reference - merge the rows into ONE table

Context: {continuity_hint.context_note or 'Table continues from previous extraction'}

"""
    
    return f"""You are extracting form fields from a document.
{continuation_block}
## CRITICAL INSTRUCTION
Extract ONLY these sections:
{sections_list}

IGNORE all other sections. They will be extracted in separate batches.

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
Return a ChunkExtraction object with ONLY the listed sections.

## CRITICAL
- Extract EVERY field - missing fields = data loss
- When in doubt, include it
- Verbatim extraction - do not summarize"""


def build_continuation_prompt(context: 'ChunkContext') -> str:
    """Build a context-aware continuation prompt."""
    base = f"""You are continuing extraction from a large document.
Extract ONLY sections present in this chunk.

{VERBATIM_EXTRACTION_RULES}

{TABLE_HANDLING_RULES}

{FIELD_TYPE_RULES}

{FIELD_PROPERTIES_RULES}

{PLACEHOLDER_RULES}

RULES:
- Section IDs will be renumbered globally
- Do NOT include document metadata (already extracted)
- Focus only on THIS chunk's content"""

    context_parts = ["\n\n## CONTEXT FROM PREVIOUS CHUNKS:"]
    
    if context.last_section_title:
        context_parts.append(f"- Last section: \"{context.last_section_title}\"")
    
    if context.total_sections_so_far > 0:
        context_parts.append(f"- Sections so far: {context.total_sections_so_far}")
    
    context_parts.append("\n## INSTRUCTIONS:")
    
    if not context.section_was_complete:
        context_parts.append(f"""
 PREVIOUS SECTION WAS CUT OFF.
If this chunk starts without a header, content belongs to: "{context.last_section_title}"
Create a section with the SAME title - system will merge automatically.""")
    
    if context.table_in_progress:
        context_parts.append(f"""
 TABLE WAS CUT OFF.
Table: "{context.table_name_in_progress or 'Unknown'}"
If you see rows starting with | without headers, these are CONTINUATION ROWS.""")
    
    if context.section_was_complete:
        context_parts.append("✓ Previous section complete. Start fresh with new headers.")
    
    return base + "\n".join(context_parts)


SYSTEM_PROMPT = build_system_prompt()