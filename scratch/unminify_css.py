import re
from pathlib import Path

SRC_CSS_PATH = Path("frontend/dist/assets/app.css")
DEST_CSS_PATH = Path("frontend/src/styles/App.css")

def unminify():
    if not SRC_CSS_PATH.is_file():
        print(f"Error: source CSS file not found at {SRC_CSS_PATH}")
        return

    print("Unminifying compiled CSS...")
    minified = SRC_CSS_PATH.read_text(encoding="utf-8")
    
    # Simple CSS unminifier/formatter
    # Add newlines after }
    formatted = minified.replace("}", "}\n\n")
    # Add newlines after { and add tab indentation
    formatted = formatted.replace("{", " {\n  ")
    # Add newlines after ; and add indentation
    formatted = formatted.replace(";", ";\n  ")
    # Clean up empty lines and trailing spaces
    formatted = re.sub(r";\s*\n\s*}", ";\n}", formatted)
    formatted = re.sub(r"{\s*\n\s*}", "{}", formatted)
    formatted = formatted.replace("  \n", "")
    
    # Fix nested media query indentations
    lines = formatted.splitlines()
    indent_level = 0
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue
            
        if line.endswith("}"):
            indent_level = max(0, indent_level - 1)
            
        prefix = "  " * indent_level
        cleaned_lines.append(prefix + line)
        
        if line.endswith("{") or line.endswith(" {"):
            indent_level += 1
            
    unminified_css = "\n".join(cleaned_lines)
    
    # Write to destination
    DEST_CSS_PATH.write_text(unminified_css, encoding="utf-8")
    print(f"Successfully unminified and saved {len(cleaned_lines)} lines to {DEST_CSS_PATH}")

if __name__ == "__main__":
    unminify()
