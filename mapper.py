
from typing import Dict, List, Any, Optional
from schemas import FormDocument, FormSection, FormField, FormTable

# Patterns that indicate a column should be a radio button
RADIO_COLUMN_PATTERNS = [
    "yes", "no", "sometimes", "unsure", "maybe", "n/a", "na",
    "agree", "disagree", "neutral",
    "always", "never", "often", "rarely",
    "true", "false",
    "correct", "incorrect",
    "completed", "not completed", "in progress",
    "satisfactory", "unsatisfactory",
    "pass", "fail",
    "met", "not met", "not yet met",
    "✓", "✗", "☐", "⬜", "☑", "□", "■"
]

# Patterns that indicate the first/main content column (not interactive)
CONTENT_COLUMN_PATTERNS = [
    "task", "item", "question", "statement", "description", "activity",
    "skill", "criteria", "element", "you can", "i can", "action",
    "prompt", "topic", "subject", "name", "details"
]


def is_interactive_column(title: str) -> bool:
    """Check if a column title indicates it's an interactive (radio/checkbox) column."""
    if not title:
        return False
    title_lower = title.lower().strip()
    
    # Check exact matches first
    for pattern in RADIO_COLUMN_PATTERNS:
        if title_lower == pattern:
            return True
    
    # Check if it's a very short title (likely Yes/No type)
    if len(title_lower) <= 10 and title_lower in RADIO_COLUMN_PATTERNS:
        return True
    
    return False


def is_content_column(title: str) -> bool:
    """Check if a column title indicates it's the main content column."""
    if not title:
        return False
    title_lower = title.lower().strip()
    
    for pattern in CONTENT_COLUMN_PATTERNS:
        if pattern in title_lower:
            return True
    
    return False


def detect_radio_group(columns: List[Dict]) -> Optional[str]:
    """Detect if columns form a radio group and return group name."""
    interactive_cols = [c for c in columns if is_interactive_column(c.get("title", ""))]
    
    if len(interactive_cols) >= 2:
        # Common patterns
        titles = [c.get("title", "").lower() for c in interactive_cols]
        
        if "yes" in titles and "no" in titles:
            return "yes_no"
        if "yes" in titles and "sometimes" in titles:
            return "frequency"
        if "agree" in titles and "disagree" in titles:
            return "agreement"
        if "met" in titles or "not met" in titles:
            return "competency"
        
        return "answer"  # Default group name
    
    return None


def transform_table_to_client(table: FormTable, table_counter: int) -> Dict[str, Any]:
    """Transform internal FormTable to client table format."""
    
    # Build columns with proper keys
    columns = []
    headers = table.headers or []
    
    # Detect if this table has interactive columns
    radio_group = None
    temp_cols = []
    
    for i, header in enumerate(headers):
        col = {
            "key": f"col_{i + 1}",
            "title": header,
            "align": "center" if is_interactive_column(header) else "left"
        }
        temp_cols.append((col, header))
    
    # Check for radio group pattern
    interactive_headers = [h for _, h in temp_cols if is_interactive_column(h)]
    if len(interactive_headers) >= 2:
        radio_group = detect_radio_group([{"title": h} for h in interactive_headers])
    

    col_empty_counts = {f"col_{i+1}": 0 for i in range(len(headers))}
    col_filled_counts = {f"col_{i+1}": 0 for i in range(len(headers))}
    
    for row in table.rows or []:
        cells = row.cells or []
        for i, cell in enumerate(cells):
            col_key = f"col_{i + 1}"
            if col_key in col_empty_counts:
                if cell.value and cell.value.strip():
                    col_filled_counts[col_key] += 1
                else:
                    col_empty_counts[col_key] += 1
    

    editable_cols = set()
    for i, header in enumerate(headers):
        col_key = f"col_{i + 1}"
        has_empties = col_empty_counts.get(col_key, 0) > 0
        has_filled = col_filled_counts.get(col_key, 0) > 0
        is_first_col = (i == 0)
        

        if has_empties and (not is_first_col or (has_filled and has_empties)):
            # Don't mark interactive columns as editable (they get radio buttons)
            if not is_interactive_column(header):
                editable_cols.add(col_key)
    
    # Now build final columns with type info
    for col, header in temp_cols:
        if radio_group and is_interactive_column(header):
            col["type"] = "radio"
            col["radioGroup"] = radio_group
            col["value"] = header.lower().replace(" ", "_")
        elif col["key"] in editable_cols:
            col["editable"] = True
        columns.append(col)
    
    # Build rows - for interactive tables, only include content columns in row data
    rows = []
    for row in table.rows or []:
        row_data = {}
        cells = row.cells or []
        
        for i, cell in enumerate(cells):
            col_key = f"col_{i + 1}"
            header = headers[i] if i < len(headers) else ""
            

            if radio_group and is_interactive_column(header):
                # Only include if there's actual checked value (non-empty)
                if cell.value and cell.value.strip():
                    row_data[col_key] = cell.value
                # Otherwise skip - frontend renders radio button
            else:
                # Content column - always include
                row_data[col_key] = cell.value if cell.value else ""
        
        rows.append(row_data)
    
    return {
        "fieldName": f"table_{table_counter}",
        "fieldType": "table",
        "label": table.title or f"Table {table_counter}",
        "required": False,
        "table": {
            "columns": columns,
            "rows": rows
        }
    }


def transform_field_to_client(field: FormField, field_counter: int) -> Dict[str, Any]:
    """Transform internal FormField to client field format."""
    
    client_field = {
        "fieldName": f"field_{field_counter}",
        "fieldType": field.type or "text",
        "label": field.label or "",
        "required": field.required or False
    }
    
    # Add optional properties if present
    if field.placeholder:
        client_field["placeholder"] = field.placeholder
    
    if field.options:
        client_field["options"] = field.options
    
    if field.value:
        client_field["value"] = field.value
    
    if field.validation:
        # Ensure validation is JSON-serializable
        client_field["validation"] = field.validation.model_dump() if hasattr(field.validation, "model_dump") else field.validation
    
    if field.depends_on:
        client_field["dependsOn"] = field.depends_on
    
    if field.show_when:
        client_field["showWhen"] = field.show_when
    
    if field.group:
        client_field["group"] = field.group
    
    if field.group_layout:
        client_field["groupLayout"] = field.group_layout
    
    return client_field


def transform_section_to_client(
    section: FormSection, 
    section_counter: int,
    field_counter: int,
    table_counter: int
) -> tuple[Dict[str, Any], int, int]:

    
    fields = []
    
    for item in section.content or []:
        if isinstance(item, FormTable):
            table_counter += 1
            fields.append(transform_table_to_client(item, table_counter))
        elif isinstance(item, FormField):
            field_counter += 1
            fields.append(transform_field_to_client(item, field_counter))
        elif isinstance(item, dict):
            # Handle dict items (might be table or field)
            if item.get("type") == "table" or "headers" in item or "rows" in item:
                table_counter += 1
                # Convert dict to FormTable-like structure
                table = FormTable(
                    headers=item.get("headers", []),
                    rows=item.get("rows", []),
                    title=item.get("title") or item.get("label", "")
                )
                fields.append(transform_table_to_client(table, table_counter))
            else:
                field_counter += 1
                field = FormField(**item)
                fields.append(transform_field_to_client(field, field_counter))
    
    client_section = {
        "section": f"section_{section_counter}",
        "sectionTitle": section.title or f"Section {section_counter}",
        "fields": fields
    }
    
    # Add optional section properties
    if section.filled_by:
        client_section["filledBy"] = section.filled_by
    
    if section.instructions:
        client_section["instructions"] = section.instructions
    
    return client_section, field_counter, table_counter


def transform_to_client_format(doc: FormDocument) -> Dict[str, Any]:

    # Initialize counters
    field_counter = 0
    table_counter = 0
    section_counter = 0
    
    # Transform sections
    form_structure = []
    for section in doc.sections or []:
        section_counter += 1
        client_section, field_counter, table_counter = transform_section_to_client(
            section, section_counter, field_counter, table_counter
        )
        form_structure.append(client_section)
    
    # Build client document - access metadata from doc.metadata
    meta = doc.metadata or {}
    if hasattr(meta, 'model_dump'):
        meta = meta.model_dump()
    elif not isinstance(meta, dict):
        meta = {}
    
    client_doc = {
        "metadata": {
            "document_name": meta.get("document_name") or doc.name or "",
            "version": meta.get("version") or "",
            "organization": meta.get("organization") or "",
            "cricos_code": meta.get("cricos_code") or "",
            "provider_code": meta.get("provider_code") or "",
            "abn": meta.get("abn") or ""
        },
        "name": doc.name or "",
        "description": doc.description or "",
        "filledBy": doc.filled_by or "user",
        "formStructure": form_structure
    }
    
    return client_doc


def transform_to_client_dict(doc) -> Dict[str, Any]:

    if isinstance(doc, dict):
        # Convert dict back to FormDocument for processing
        doc = FormDocument(**doc)
    
    return transform_to_client_format(doc)


def transform_dict_to_client(data: Dict[str, Any]) -> Dict[str, Any]:

    doc = FormDocument(**data)
    return transform_to_client_format(doc)
