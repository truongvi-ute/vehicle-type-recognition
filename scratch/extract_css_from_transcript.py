import json
import re
from pathlib import Path

TRANSCRIPT_PATH = Path(r"C:\Users\Admin\.gemini\antigravity\brain\e7c93550-5295-4774-b219-bdbe0c8dc2c8\.system_generated\logs\transcript_full.jsonl")

def search_transcript():
    if not TRANSCRIPT_PATH.is_file():
        print(f"Error: transcript file not found at {TRANSCRIPT_PATH}")
        return

    print("Searching transcript for dashboard CSS segments...")
    
    with TRANSCRIPT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
                content = step.get("content", "")
                # We search for dashboard classes like .exp-section or .experiment-dashboard
                if ".exp-section" in content and ".dashboardContent" in content:
                    print(f"Found a matching step {step.get('step_index')} (type: {step.get('type')})")
                    # If this is a tool response (usually has a field output or we check content)
                    # Let's extract all CSS code from it!
                    # Often it has line numbers or is raw text. Let's see if we can find it.
                    out_path = Path(f"scratch/recovered_step_{step.get('step_index')}.css")
                    
                    # Try to see if it has line numbers like "123: .classname {"
                    # We can use regex to extract
                    lines = content.splitlines()
                    clean_lines = []
                    has_line_numbers = False
                    
                    # Check if many lines start with digits followed by colon
                    numbered_lines = 0
                    for l in lines[:20]:
                        if re.match(r"^\s*\d+:", l):
                            numbered_lines += 1
                    if numbered_lines > 5:
                        has_line_numbers = True
                        
                    if has_line_numbers:
                        print("Step has line numbers. Stripping them...")
                        for l in lines:
                            m = re.match(r"^\s*\d+:\s?(.*)", l)
                            if m:
                                clean_lines.append(m.group(1))
                            else:
                                if not any(x in l for x in ["File Path:", "Total Lines:", "Total Bytes:", "Showing lines"]):
                                    clean_lines.append(l)
                    else:
                        print("Step has raw text. Copying directly...")
                        clean_lines = lines
                        
                    recovered = "\n".join(clean_lines)
                    out_path.write_text(recovered, encoding="utf-8")
                    print(f"Saved recovered CSS to {out_path} ({len(clean_lines)} lines)")
            except Exception as e:
                print("Error parsing line:", e)

if __name__ == "__main__":
    search_transcript();
