import json
from pathlib import Path

SESSION_ID = "58bfa905-06a5-453a-b00f-ad6c0ade2271"
TRANSCRIPT_PATH = Path(fr"C:\Users\Admin\.gemini\antigravity\brain\{SESSION_ID}\.system_generated\logs\transcript.jsonl")

def extract():
    if not TRANSCRIPT_PATH.is_file():
        print(f"Error: transcript file not found at {TRANSCRIPT_PATH}")
        return

    print(f"Extracting all App.css writes from session {SESSION_ID}...")
    
    with TRANSCRIPT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
                if "tool_calls" in step:
                    for tc in step["tool_calls"]:
                        args = tc.get("args", {})
                        target = args.get("TargetFile", "")
                        if "App.css" in target:
                            print(f"Step {step.get('step_index')}: tool={tc.get('name')}")
                            for k, v in args.items():
                                if k in ["CodeContent", "ReplacementContent"]:
                                    print(f"  {k} (length={len(v)})")
                                    # Save to scratch
                                    out_path = Path(f"scratch/recovered_prev_{step.get('step_index')}_{k}.css")
                                    out_path.write_text(v, encoding="utf-8")
                                    print(f"    Saved to {out_path}")
                                elif k not in ["TargetFile", "toolAction", "toolSummary"]:
                                    print(f"  {k}: {v}")
            except Exception as e:
                pass

if __name__ == "__main__":
    extract()
