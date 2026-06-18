import json
from pathlib import Path

TRANSCRIPT_PATH = Path(r"C:\Users\Admin\.gemini\antigravity\brain\e7c93550-5295-4774-b219-bdbe0c8dc2c8\.system_generated\logs\transcript_full.jsonl")

def inspect_step():
    with TRANSCRIPT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
                if step.get("step_index") == 663:
                    print(f"Step 663 metadata: type={step.get('type')}, status={step.get('status')}")
                    # Print tool calls or content
                    if "tool_calls" in step:
                        print("Tool calls:")
                        for tc in step["tool_calls"]:
                            print(f"  Name: {tc.get('name')}")
                            # If it's replace_file_content or write_to_file
                            args = tc.get("args", {})
                            print(f"  Args: {list(args.keys())}")
                            if "CodeContent" in args:
                                print("  CodeContent length:", len(args["CodeContent"]))
                            if "ReplacementContent" in args:
                                print("  ReplacementContent length:", len(args["ReplacementContent"]))
                    # Check for tool results or other fields
                    for k, v in step.items():
                        if k not in ["content", "tool_calls"]:
                            print(f"  {k}: {v}")
                    content = step.get("content", "")
                    if content:
                        print(f"  Content length: {len(content)}")
                        print("  Snippet:", content[:300])
                    # If this step was a tool response, let's look for it
            except Exception as e:
                print(e)

if __name__ == "__main__":
    inspect_step()
