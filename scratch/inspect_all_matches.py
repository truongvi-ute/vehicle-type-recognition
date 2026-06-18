import json
from pathlib import Path

TRANSCRIPT_PATH = Path(r"C:\Users\Admin\.gemini\antigravity\brain\e7c93550-5295-4774-b219-bdbe0c8dc2c8\.system_generated\logs\transcript_full.jsonl")

def inspect_all():
    target_steps = [661, 758, 840, 898, 923, 927, 932, 934, 951, 961, 962, 963]
    with TRANSCRIPT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
                step_idx = step.get("step_index")
                if step_idx in target_steps:
                    print(f"\n=== Step {step_idx} (type: {step.get('type')}) ===")
                    tool_name = ""
                    if "tool_calls" in step:
                        for tc in step["tool_calls"]:
                            print(f"  Tool Call: {tc.get('name')}")
                            args = tc.get("args", {})
                            print(f"    Args: {list(args.keys())}")
                            if "CommandLine" in args:
                                print(f"      CommandLine: {args['CommandLine']}")
                            if "TargetFile" in args:
                                print(f"      TargetFile: {args['TargetFile']}")
                    content = step.get("content", "")
                    if content:
                        print(f"  Content length: {len(content)}")
                        print("  Content snippet:")
                        print(content[:500] + "\n...")
            except Exception as e:
                pass

if __name__ == "__main__":
    inspect_all()
