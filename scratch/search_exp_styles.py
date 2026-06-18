import json
import os
from pathlib import Path

BRAIN_DIR = Path(r"C:\Users\Admin\.gemini\antigravity\brain")

def search():
    print("Searching all transcripts for .exp- styles...")
    matches = []
    for item in BRAIN_DIR.iterdir():
        if item.is_dir() and item.name != "tempmediaStorage":
            logs_dir = item / ".system_generated" / "logs"
            for log_file in ["transcript_full.jsonl", "transcript.jsonl"]:
                path = logs_dir / log_file
                if path.is_file():
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            for line in f:
                                if "exp-compare" in line or "exp-section" in line:
                                    # We found a match!
                                    step = json.loads(line)
                                    tool_name = ""
                                    if "tool_calls" in step:
                                        tool_name = ", ".join([tc.get("name", "") for tc in step["tool_calls"]])
                                    print(f"  Session: {item.name}, Log: {log_file}, Step: {step.get('step_index')}, Type: {step.get('type')}, Tool: {tool_name}")
                                    # Let's save the match details
                                    matches.append({
                                        "session": item.name,
                                        "log": log_file,
                                        "step": step.get("step_index"),
                                        "line": line
                                    })
                    except Exception as e:
                        pass
    print(f"Total matches found: {len(matches)}")
    # Save the matches list to a json file
    Path("scratch/matches_exp.json").write_text(json.dumps([{
        "session": m["session"],
        "log": m["log"],
        "step": m["step"]
    } for m in matches], indent=2), encoding="utf-8")

if __name__ == "__main__":
    search()
