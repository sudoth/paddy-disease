import subprocess
from pathlib import Path


def get_git_commit(repo_root: Path) -> str | None:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip()
