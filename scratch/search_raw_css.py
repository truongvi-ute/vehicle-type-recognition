import json
from pathlib import Path

TRANSCRIPT_PATH = Path(r"C:\Users\Admin\.gemini\antigravity\brain\e7c93550-5295-4774-b219-bdbe0c8dc2c8\.system_generated\logs\transcript_full.jsonl")

def search():
    with TRANSCRIPT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
                content = step.get("content", "")
                if "exp-section" in content:
                    print(f"Step {step.get('step_index')}: type={step.get('type')}, length={len(content)}")
                    # Write it out
                    out_path = Path(f"scratch/raw_match_{step.get('step_index')}.txt")
                    out_path.write_text(content, encoding="utf-8")
                    print(f"  Saved raw text to {out_path}")
            except Exception as e:
                pass

if __name__ == "__main__":
    search()
