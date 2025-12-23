from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field


# Table column definition
class TableColumn(BaseModel):
    """A column in a client table."""
    key: str = Field(..., description="Column key: col_1, col_2, etc.")
    title: str = Field(..., description="Column header text")
    align: Literal["left", "center", "right"] = Field("left", description="Text alignment")
    type: Optional[Literal["text", "radio", "checkbox"]] = Field(None, description="Column type: text (default), radio, or checkbox")
    radioGroup: Optional[str] = Field(None, description="Radio group name (for radio type)")
    value: Optional[str] = Field(None, description="Value for this radio/checkbox option")


# Table structure for client
class ClientTable(BaseModel):
    """Table structure for client consumption."""
    columns: List[TableColumn] = Field(..., description="Column definitions with keys")
    rows: List[Dict[str, str]] = Field(..., description="Row data as {col_1: value, col_2: value}")


# Field types for client
ClientFieldType = Literal[
    "label",
    "text",
    "textarea", 
    "number",
    "date",
    "checkbox",
    "checkbox_group",
    "radio",
    "select",
    "signature",
    "table"
]


# Client field (non-table)
class ClientField(BaseModel):
    """A field in the client format."""
    fieldName: str = Field(..., description="Auto-generated ID: field_1, field_2, or table_1")
    fieldType: ClientFieldType = Field(..., description="Field type for UI rendering")
    label: str = Field(..., description="Display label text")
    required: bool = Field(False, description="Whether field is required")
    
    # Optional properties based on field type
    options: Optional[List[str]] = Field(None, description="Options for radio/select/checkbox_group")
    placeholder: Optional[str] = Field(None, description="Placeholder text for input fields")
    value: Optional[str] = Field(None, description="Pre-filled value if any")
    
    # Table data (only present when fieldType == "table")
    table: Optional[ClientTable] = Field(None, description="Table data when fieldType is 'table'")


# Client section
class ClientSection(BaseModel):
    """A section in the client format."""
    section: str = Field(..., description="Section ID: section_1, section_2, etc.")
    sectionTitle: str = Field(..., description="Section title/heading")
    fields: List[ClientField] = Field(..., description="Fields in this section")


# Client document metadata
class ClientMetadata(BaseModel):
    """Metadata for client document."""
    document_name: Optional[str] = None
    version: Optional[str] = None
    organization: Optional[str] = None
    cricos_code: Optional[str] = None
    provider_code: Optional[str] = None
    abn: Optional[str] = None


# Complete client document
class ClientDocument(BaseModel):
    """Complete document in client format."""
    metadata: Optional[ClientMetadata] = Field(None, description="Document metadata")
    name: str = Field(..., description="Full document name")
    description: str = Field(..., description="Document purpose description")
    stepNumber: Optional[int] = Field(None, description="Step number in workflow")
    filledBy: str = Field(..., description="Who fills: user, assessor, thirdparty, mixed")
    formStructure: List[ClientSection] = Field(..., description="All sections with fields")