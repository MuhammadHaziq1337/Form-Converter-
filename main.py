import argparse
import sys
import json
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from parser import convert_to_markdown
from extractor import extract_document_data

def process_document(file_path: str, verbose: bool = False) -> dict:

    if verbose:
        print(f"[1/3] Parsing document: {file_path}")
    
    # Step 1: Convert document to Markdown
    markdown_content = convert_to_markdown(file_path)
    
    if verbose:
        print(f"[2/3] Document converted to Markdown ({len(markdown_content)} chars)")
        print("-" * 50)
        # Show first 500 chars of markdown for debugging
        print(markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)
        print("-" * 50)
    
    # Step 2: Extract structured data using LLM
    if verbose:
        model_name = os.getenv("OPENAI_MODEL", "gpt-5")
        print(f"[3/3] Extracting data with {model_name}...")
    
    document_data = extract_document_data(markdown_content)
    
    # Step 3: Return validated data as dict
    return document_data.model_dump()


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Convert PDF/DOCX documents to structured JSON data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py assessment.pdf
    python main.py form.docx --verbose
        """
    )
    
    parser.add_argument(
        "file_path",
        help="Path to the document file (PDF or DOCX)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print intermediate processing steps"
    )
    
    args = parser.parse_args()
    
    try:
        # Process the document
        result = process_document(args.file_path, verbose=args.verbose)
        
        # Format as JSON
        output_json = json.dumps(result, indent=2, ensure_ascii=False)
        
        # Generate output filename (same name as input, but .json extension)
        input_path = Path(args.file_path)
        output_path = input_path.with_suffix(".json")
        
        # Save JSON to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_json)
        
        print(f"✓ Results saved to: {output_path}")
            
    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

