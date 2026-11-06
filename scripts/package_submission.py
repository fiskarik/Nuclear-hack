"""Package only declared project/data/run paths, never credentials or Git metadata."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(data, run, output, holdout=None):
    data, run, output = Path(data).resolve(), Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    required = [data / "manifest.json", run / "best.pt", run / "training.json", run / "benchmark.json",
                run / "result/doe_phase.png", run / "result/metrics.json", ROOT / "notebooks/fresnel_doe_kaggle.ipynb"]
    if holdout:
        required.append(Path(holdout) / "manifest.json")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    entries = []
    for name in ("README.md", "pyproject.toml", "requirements.txt", "requirements-dev.txt", ".gitignore"):
        entries.append((ROOT / name, "source/" + name))
    for folder in ("doe", "tests", "scripts", "docs", "notebooks"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                entries.append((path, "source/" + str(path.relative_to(ROOT))))
    for prefix, directory in (("data", data), ("run", run), ("independent_holdout", Path(holdout).resolve() if holdout else None)):
        if directory is None:
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path != output:
                entries.append((path, prefix + "/" + str(path.relative_to(directory))))
    if len({name for _, name in entries}) != len(entries):
        raise ValueError("duplicate archive paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in entries:
            archive.write(path, name)
    with zipfile.ZipFile(output) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"Archive CRC failure: {corrupt}")
    print(f"{output}: {len(entries)} files, {output.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--holdout")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    package(**vars(args))


if __name__ == "__main__":
    main()
