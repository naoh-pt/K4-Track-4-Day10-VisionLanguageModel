"""Đóng gói source để upload Colab, không cần commit/push."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    files = []
    for directory, suffix in (("src", ".py"), ("scripts", ".py"), ("tests", ".py"), ("configs", ".yaml")):
        files.extend((root / directory).rglob(f"*{suffix}"))
    files.extend(p for p in (root / "report").rglob("*") if p.is_file() and p.suffix in {".json", ".png", ".txt"})
    files.extend(root / name for name in ("README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md",
                 "IMPLEMENTATION_PLAN.md", "COLAB_RUN_GUIDE.md", "requirements-colab.txt",
                 "pytest.ini", ".gitignore", "lab_day10.ipynb"))
    destination = root / "work/colab_bundle.zip"
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    print(destination)


if __name__ == "__main__":
    main()
