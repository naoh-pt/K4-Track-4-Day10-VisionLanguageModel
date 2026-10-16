"""Đóng gói source để upload Colab, không cần commit/push."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="work/colab_bundle.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = []
    for directory, suffix in (("src", ".py"), ("scripts", ".py"), ("tests", ".py"), ("configs", ".yaml")):
        files.extend((root / directory).rglob(f"*{suffix}"))
    files.extend(p for p in (root / "report").rglob("*") if p.is_file() and p.suffix in {".json", ".png", ".txt"})
    files.extend(root / name for name in ("README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md",
                 "requirements-colab.txt",
                 "pytest.ini", ".gitignore", "lab_day10.ipynb"))
    destination = (root / args.out).resolve()
    if not destination.is_relative_to((root / "work").resolve()):
        raise ValueError("Bundle phải lưu trong work/ để không vô tình commit ZIP.")
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    print(destination)


if __name__ == "__main__":
    main()
