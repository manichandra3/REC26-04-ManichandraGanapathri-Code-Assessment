#!/usr/bin/env python3
"""Cross-platform runner for the Plain-Vanilla RAG pipeline.

Usage:
    python scripts/run.py setup           # create venv + install deps
    python scripts/run.py ingest          # parse PDFs and populate ChromaDB
    python scripts/run.py query <question>  # ask a question
    python scripts/run.py evaluate        # run groundedness evaluation
    python scripts/run.py all             # setup → ingest → interactive query
"""

import os
import sys
import subprocess
import platform
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pip() -> str:
    if platform.system() == "Windows":
        return os.path.join(ROOT, "venv", "Scripts", "pip.exe")
    return os.path.join(ROOT, "venv", "bin", "pip")


def _python() -> str:
    if platform.system() == "Windows":
        return os.path.join(ROOT, "venv", "Scripts", "python.exe")
    return os.path.join(ROOT, "venv", "bin", "python")


def step_setup():
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        print("ERROR: .env not found. Copy .env.example to .env and add your GEMINI_API_KEY.")
        sys.exit(1)

    if not os.path.exists(os.path.join(ROOT, "venv")):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", os.path.join(ROOT, "venv")], check=True)
        print("Installing dependencies...")
        subprocess.run([_pip(), "install", "-r", os.path.join(ROOT, "requirements.txt")], check=True)
        print("Setup complete.")
    else:
        print("Virtual environment already exists. Use 'reinstall' to rebuild.")


def step_ingest():
    print("=== Ingest ===")
    subprocess.run([_python(), os.path.join(ROOT, "ingest.py")], check=True)


def step_query(question: str | None):
    print("=== Query ===")
    cmd = [_python(), os.path.join(ROOT, "query.py")]
    if question:
        cmd.append(question)
    subprocess.run(cmd, check=True)


def step_evaluate():
    print("=== Evaluate ===")
    subprocess.run([_python(), os.path.join(ROOT, "evaluate.py")], check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument("step", choices=["setup", "ingest", "query", "evaluate", "all"],
                        help="Pipeline step to run")
    parser.add_argument("question", nargs="*", help="Question (required for 'query', optional for 'all')")
    args = parser.parse_args()

    if args.step == "setup":
        step_setup()
    elif args.step == "ingest":
        step_ingest()
    elif args.step == "query":
        if not args.question:
            parser.error("the 'query' step requires a question argument")
        step_query(" ".join(args.question))
    elif args.step == "evaluate":
        step_evaluate()
    elif args.step == "all":
        step_setup()
        step_ingest()
        if args.question:
            step_query(" ".join(args.question))
        else:
            step_query(input("Enter your question: "))


if __name__ == "__main__":
    main()
