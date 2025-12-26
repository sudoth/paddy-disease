from pathlib import Path


def main() -> None:
    """
    Entry point for the project.
    """
    repo_root = Path(__file__).parents[2]
    print(f"entry point is working, repo root is {repo_root}")


if __name__ == "__main__":
    main()
