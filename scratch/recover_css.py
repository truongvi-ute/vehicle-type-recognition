import json
from pathlib import Path

TRANSCRIPT_PATH = Path(r"C:\Users\Admin\.gemini\antigravity\brain\4f0a13aa-5196-480a-8c0f-c0674d30e3e2\.system_generated\logs\transcript_full.jsonl")

def search_css():
    global TRANSCRIPT_PATH
    if not TRANSCRIPT_PATH.is_file():
        print(f"Error: transcript file not found at {TRANSCRIPT_PATH}")
        alt_path = Path(r"C:\Users\Admin\.gemini\antigravity\brain\4f0a13aa-5196-480a-8c0f-c0674d30e3e2\.system_generated\logs\transcript.jsonl")
        if alt_path.is_file():
            print(f"Falling back to non-full transcript at {alt_path}")
            TRANSCRIPT_PATH = alt_path
        else:
            print("No previous session transcript found.")
            return

    print(f"Searching transcript {TRANSCRIPT_PATH} for App.css edits...")
    last_content = None
    last_step = None
    
    with TRANSCRIPT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
                if "tool_calls" in step:
                    for tc in step["tool_calls"]:
                        args = tc.get("args", {})
                        target = args.get("TargetFile", "")
                        if "App.css" in target:
                            content = args.get("CodeContent")
                            if content:
                                last_content = content
                                last_step = step.get("step_index")
            except Exception as e:
                pass
                
    if last_content:
        print(f"Found complete content written to App.css at step {last_step}.")
        out_path = Path("scratch/recovered_app.css")
        out_path.write_text(last_content, encoding="utf-8")
        print(f"Successfully recovered to {out_path}")
    else:
        print("No complete write found in previous transcript. Checking outputs...")
        find_view_outputs()

def find_view_outputs():
    with TRANSCRIPT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
                content = step.get("content", "")
                if "Total Lines: 1854" in content or "exp-error-list" in content:
                    print(f"Found match in step {step.get('step_index')}")
                    Path(f"scratch/step_{step.get('step_index')}.json").write_text(json.dumps(step, indent=2), encoding="utf-8")
            except Exception as e:
                pass

if __name__ == "__main__":
    search_css()
