import json
import os
from pathlib import Path

BRAIN_DIR = Path(r"C:\Users\Admin\.gemini\antigravity\brain")

def search_all():
    print(f"Searching all subfolders in {BRAIN_DIR} for App.css edits...")
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
                                if "App.css" in line:
                                    step = json.loads(line)
                                    if "tool_calls" in step:
                                        for tc in step["tool_calls"]:
                                            args = tc.get("args", {})
                                            target = args.get("TargetFile", "")
                                            if "App.css" in target:
                                                content = args.get("CodeContent") or args.get("ReplacementContent")
                                                if content and len(content) > 500:
                                                    print(f"  Session: {item.name}, Step: {step.get('step_index')}, Tool: {tc.get('name')}, Content Length: {len(content)}")
                                                    matches.append({
                                                        "session": item.name,
                                                        "step": step.get("step_index"),
                                                        "tool": tc.get("name"),
                                                        "content": content
                                                    })
                    except Exception as e:
                        pass
                        
    print(f"Found {len(matches)} large writes to App.css.")
    # If we found any matches, let's write out the largest one to recovered_css.css
    if matches:
        # Sort by content length
        matches.sort(key=lambda x: len(x["content"]), reverse=True)
        best = matches[0]
        print(f"Largest write: Session {best['session']}, Step {best['step']}, length {len(best['content'])}")
        out_path = Path("scratch/recovered_css.css")
        out_path.write_text(best["content"], encoding="utf-8")
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    search_all()
