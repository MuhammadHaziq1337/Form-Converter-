# CIO Form Converter

AI-powered document extraction system that converts complex DOCX forms into structured JSON with field-level accuracy.

## Features

- **Zero-Gap Streaming**: Physical chunking ensures 100% document coverage with no data loss
- **Smart Extraction**: Identifies form fields, tables, checkboxes, signatures, and conditional logic
- **Hybrid Model Support**: Automatically selects optimal AI model based on document size
- **State Injection**: Maintains context across chunks for accurate section continuity
- **Table Boundary Detection**: Never splits tables mid-row

## Requirements

- Python 3.10+
- OpenAI API key

## Installation

```bash
# Clone repository
git clone <repository-url>
cd CIO-FormConverter

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo OPENAI_API_KEY=your_api_key_here > .env
```

## Usage

### Web Interface

```bash
python server.py
```

Open `http://localhost:8080` and upload a DOCX file.

### Command Line

```python
from extractor import extract_form_data

markdown = "..."  # Your document markdown
result = extract_form_data(markdown)
print(result.document)
```

## Configuration

Edit `.env`:

```bash
OPENAI_MODEL=gpt-5             # Default model (gpt-5 for best accuracy)
HYBRID_MODEL=true              # Auto-switch to GPT-5 for large docs
EXTRACTOR_DEBUG=false          # Save debug markdown files
```

## Architecture

- **Single-Pass**: Small documents (<150k tokens) extract in one call
- **Streaming Mode**: Large documents split into physical chunks with state continuity
- **Retry Logic**: Automatically retries on low coverage or rate limits

## Output Schema

```json
{
  "metadata": {...},
  "name": "Document Name",
  "sections": [
    {
      "title": "Section 1",
      "content": [
        {"type": "text", "label": "Name", "editable": true},
        {"type": "table", "headers": [...], "rows": [...]}
      ]
    }
  ]
}
```

