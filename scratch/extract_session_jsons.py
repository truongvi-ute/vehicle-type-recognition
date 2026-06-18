import json
from pathlib import Path

SESSION_ID = "4f0a13aa-5196-480a-8c0f-c0674d30e3e2"
TRANSCRIPT_PATH = Path(fr"C:\Users\Admin\.gemini\antigravity\brain\{SESSION_ID}\.system_generated\logs\transcript_full.jsonl")

def extract():
    global TRANSCRIPT_PATH
    if not TRANSCRIPT_PATH.is_file():
        print("Transcript full not found, using transcript.jsonl")
        TRANSCRIPT_PATH = Path(fr"C:\Users\Admin\.gemini\antigravity\brain\{SESSION_ID}\.system_generated\logs\transcript.jsonl")
        
    print(f"Reading from {TRANSCRIPT_PATH}...")
    with TRANSCRIPT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
                step_idx = step.get("step_index")
                if step_idx in [50, 52]:
                    print(f"Step {step_idx} (type: {step.get('type')})")
                    if "tool_calls" in step:
                        for tc in step["tool_calls"]:
                            args = tc.get("args", {})
                            print(f"  Tool: {tc.get('name')}")
                            print(f"  Keys in args: {list(args.keys())}")
                            target = args.get("TargetFile", "")
                            print(f"  TargetFile: {target}")
                            for k, v in args.items():
                                if k in ["CodeContent", "ReplacementContent"]:
                                    print(f"    Content length: {len(v)}")
                                    out_path = Path("scratch") / Path(target).name
                                    out_path.write_text(v, encoding="utf-8")
                                    print(f"    Saved to {out_path}")
            except Exception as e:
                print("Error:", e)

if __name__ == "__main__":
    extract()
