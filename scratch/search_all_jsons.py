import json
import os
from pathlib import Path

BRAIN_DIR = Path(r"C:\Users\Admin\.gemini\antigravity\brain")

def search_jsons():
    print(f"Searching all sessions in {BRAIN_DIR}...")
    for item in BRAIN_DIR.iterdir():
        if item.is_dir() and item.name != "tempmediaStorage":
            logs_dir = item / ".system_generated" / "logs"
            for log_file in ["transcript_full.jsonl", "transcript.jsonl"]:
                path = logs_dir / log_file
                if path.is_file():
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            for line in f:
                                if "data_raw_original_v1_counts.json" in line or "data_cleaning_v1_counts.json" in line:
                                    step = json.loads(line)
                                    print(f"FOUND MENTION: Session {item.name}, Step {step.get('step_index')}, Log {log_file}")
                                    # Let's see if this contains write_to_file with contents
                                    if "tool_calls" in step:
                                        for tc in step["tool_calls"]:
                                            args = tc.get("args", {})
                                            target = args.get("TargetFile", "")
                                            if "counts.json" in target:
                                                code = args.get("CodeContent")
                                                if code:
                                                    print(f"  -> File write found! Target: {target}, size: {len(code)}")
                                                    out_path = Path("scratch") / Path(target).name
                                                    out_path.write_text(code, encoding="utf-8")
                                                    print(f"  -> Restored to {out_path}")
                    except Exception as e:
                        pass

if __name__ == "__main__":
    search_jsons()
