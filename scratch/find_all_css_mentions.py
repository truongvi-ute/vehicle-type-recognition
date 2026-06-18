import json
from pathlib import Path

SESSIONS = [
    Path(r"C:\Users\Admin\.gemini\antigravity\brain\e7c93550-5295-4774-b219-bdbe0c8dc2c8"),
    Path(r"C:\Users\Admin\.gemini\antigravity\brain\4f0a13aa-5196-480a-8c0f-c0674d30e3e2")
]

def search():
    for session_dir in SESSIONS:
        if not session_dir.is_dir():
            continue
        print(f"\n--- Checking session {session_dir.name} ---")
        logs_dir = session_dir / ".system_generated" / "logs"
        for log_name in ["transcript_full.jsonl", "transcript.jsonl"]:
            log_path = logs_dir / log_name
            if not log_path.is_file():
                continue
            print(f"Reading {log_name}...")
            with log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        step = json.loads(line)
                        content = str(step)
                        if "App.css" in content:
                            tool_name = ""
                            if "tool_calls" in step:
                                tool_name = ", ".join([tc.get("name", "") for tc in step["tool_calls"]])
                            print(f"  Step {step.get('step_index')} (type: {step.get('type')}, tool: {tool_name}) has App.css mention.")
                            # Check if it was a write or replace
                            if "tool_calls" in step:
                                for tc in step["tool_calls"]:
                                    args = tc.get("args", {})
                                    if "App.css" in args.get("TargetFile", ""):
                                        print(f"    -> TargetFile is App.css. Args keys: {list(args.keys())}")
                    except Exception as e:
                        pass

if __name__ == "__main__":
    search()
