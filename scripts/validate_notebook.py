"""Execute every notebook cell; quick mode validates integration, not model quality."""
import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="notebooks/fresnel_doe_kaggle.ipynb")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--quick-check", action="store_true")
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.quick_check:
        os.environ["DOE_QUICK_CHECK"] = "1"
    else:
        os.environ.pop("DOE_QUICK_CHECK", None)
    client = NotebookClient(notebook, timeout=1800, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(workdir)}})
    client.execute()
    executed = 0
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert not any(x.output_type == "error" for x in cell.outputs)
            executed += 1
    notebook.metadata["validation"] = {"quick_check": args.quick_check, "executed_code_cells": executed}
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Executed {executed} code cells; quick_check={args.quick_check}; {output}")


if __name__ == "__main__":
    main()
