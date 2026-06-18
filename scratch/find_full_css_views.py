import json
import os
from pathlib import Path

BRAIN_DIR = Path(r"C:\Users\Admin\.gemini\antigravity\brain")

def search_all_views():
    print("Searching all sessions for full App.css views...")
    for item in BRAIN_DIR.iterdir():
        if item.is_dir() and item.name != "tempmediaStorage":
            logs_dir = item / ".system_generated" / "logs"
            for log_file in ["transcript_full.jsonl", "transcript.jsonl"]:
                path = logs_dir / log_file
                if path.is_file():
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            for line in f:
                                if "styles/App.css" in line and "Total Lines: 1" in line:
                                    # Parse step
                                    step = json.loads(line)
                                    content = step.get("content", "")
                                    # Check if it has a large line count
                                    m = re.search(r"Total Lines:\s*(\d+)", content)
                                    if m:
                                        lines = int(m.group(1))
                                        if lines > 1000:
                                            print(f"FOUND VIEW in session {item.name}, step {step.get('step_index')}, lines: {lines}")
                                            out_path = Path(f"scratch/recovered_view_{item.name}_{step.get('step_index')}.css")
                                            # Extract code
                                            extract_code(content, out_path)
                    except Exception as e:
                        pass

import re

def extract_code(content, out_path):
    lines = content.splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        if line.startswith("File Path:") or line.startswith("Total Lines:") or line.startswith("Total Bytes:") or line.startswith("Showing lines"):
            continue
        parts = line.split(":", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            code_lines.append(parts[1])
            in_code = True
        elif in_code:
            code_lines.append(line)
            
    cleaned_lines = []
    for line in code_lines:
        if line.startswith(" "):
            cleaned_lines.append(line[1:])
        else:
            cleaned_lines.append(line)
            
    recovered_css = "\n".join(cleaned_lines)
    out_path.write_text(recovered_css, encoding="utf-8")
    print(f"Saved to {out_path} ({len(cleaned_lines)} lines)")

if __name__ == "__main__":
    search_all_views()
