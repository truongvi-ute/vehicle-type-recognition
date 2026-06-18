import json
from pathlib import Path

TRANSCRIPT_PATH = Path(r"C:\Users\Admin\.gemini\antigravity\brain\e7c93550-5295-4774-b219-bdbe0c8dc2c8\.system_generated\logs\transcript_full.jsonl")

def extract():
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
                            # Print keys and some lengths
                            for k, v in args.items():
                                if k in ["CodeContent", "ReplacementContent"]:
                                    print(f"  {k} (length={len(v)})")
                                    # Write it to scratch for inspection
                                    out_path = Path(f"scratch/step_{step.get('step_index')}_{k}.css")
                                    out_path.write_text(v, encoding="utf-8")
                                    print(f"    Saved to {out_path}")
                                elif k not in ["TargetFile", "toolAction", "toolSummary"]:
                                    print(f"  {k}: {v}")
            except Exception as e:
                pass

if __name__ == "__main__":
    extract()
