from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Check that the Stage 1 project setup is ready."""
    print("=== Intelligent Document QA: Environment Check ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")

    if sys.prefix == sys.base_prefix:
        print("[FAIL] Virtual environment is not active.")
        print("Activate .venv, then run this script again.")
        return 1

    print("[OK] Virtual environment is active.")

    try:
        dotenv_version = version("python-dotenv")
    except PackageNotFoundError:
        print("[FAIL] python-dotenv is not installed in this environment.")
        print("Run: python -m pip install -r requirements.txt")
        return 1

    print(f"[OK] python-dotenv version: {dotenv_version}")

    checks = {
        ".gitignore": PROJECT_ROOT / ".gitignore",
        ".env.example": PROJECT_ROOT / ".env.example",
        "requirements.txt": PROJECT_ROOT / "requirements.txt",
        "rag package": PROJECT_ROOT / "rag" / "__init__.py",
        "tests folder": PROJECT_ROOT / "tests",
        "PDF uploads folder": PROJECT_ROOT / "data" / "uploads",
        "vector-store folder": PROJECT_ROOT / "data" / "vector_store",
        "Git repository": PROJECT_ROOT / ".git",
    }

    all_checks_passed = True

    for label, path in checks.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {label}")
        if not exists:
            all_checks_passed = False

    if not all_checks_passed:
        print("Status: FAIL — fix the missing items above.")
        return 1

    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())