import json
import os
from pathlib import Path

BRAIN_DIR = Path(r"C:\Users\Admin\.gemini\antigravity\brain")

def search_all():
    print(f"Searching all subfolders in {BRAIN_DIR}...")
    for item in BRAIN_DIR.iterdir():
        if item.is_dir() and item.name != "tempmediaStorage":
            logs_dir = item / ".system_generated" / "logs"
            for log_file in ["transcript_full.jsonl", "transcript.jsonl"]:
                path = logs_dir / log_file
                if path.is_file():
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            for line in f:
                                if "pipelineGrid" in line and "modelOption" in line:
                                    step = json.loads(line)
                                    # We found it!
                                    print(f"FOUND MATCH in session {item.name}, file {log_file}, step {step.get('step_index')}")
                                    content = step.get("content", "")
                                    if "Total Lines:" in content:
                                        print(f"This step contains the file view/content! Content length: {len(content)}")
                                        out_path = Path("scratch/recovered_step_css.json")
                                        out_path.write_text(json.dumps(step, indent=2), encoding="utf-8")
                                        print(f"Wrote step data to {out_path}")
                                        extract_code(content)
                                        return
                    except Exception as e:
                        pass

def extract_code(content):
    print("Extracting code from view output...")
    lines = content.split("\n")
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
    out_path = Path("scratch/recovered_app.css")
    out_path.write_text(recovered_css, encoding="utf-8")
    print(f"Successfully recovered full CSS to {out_path} ({len(cleaned_lines)} lines)")

if __name__ == "__main__":
    search_all()
