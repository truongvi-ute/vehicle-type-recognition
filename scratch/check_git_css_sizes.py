import subprocess

def check_all_commits():
    # Get all commit hashes
    res = subprocess.run(["git", "log", "--all", "--format=%H"], capture_output=True, text=True, check=True)
    commits = res.stdout.strip().split("\n")
    
    print(f"Checking {len(commits)} commits for App.css size...")
    for commit in commits:
        # Get line count of App.css in this commit
        res_show = subprocess.run(["git", "show", f"{commit}:frontend/src/styles/App.css"], capture_output=True, text=True)
        if res_show.returncode == 0:
            lines = len(res_show.stdout.splitlines())
            # Get commit subject
            res_subj = subprocess.run(["git", "log", "-n", "1", "--format=%s", commit], capture_output=True, text=True)
            subj = res_subj.stdout.strip()
            print(f"Commit {commit[:8]} ({subj}): {lines} lines")
            if lines > 1000:
                print(f"--> FOUND LONGER CSS ({lines} lines) in commit {commit[:8]}!")
                # Let's save it
                recovered_path = f"scratch/recovered_git_{commit[:8]}.css"
                with open(recovered_path, "w", encoding="utf-8") as f:
                    f.write(res_show.stdout)
                print(f"    Saved to {recovered_path}")

if __name__ == "__main__":
    check_all_commits()
