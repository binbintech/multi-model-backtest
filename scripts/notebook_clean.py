"""Git clean filter: keep local notebook outputs, commit source-only JSON."""
import ast
import json
import re
import sys


def clean_notebook(raw):
    notebook = json.loads(raw)
    if not isinstance(notebook.get("cells"), list):
        raise ValueError("Input does not contain notebook cells")
    keywords = ("token", "password", "secret", "api_key", "apikey")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = source if isinstance(source, str) else "".join(source)
        if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})", source):
            raise ValueError(f"Possible credential in code cell {index + 1}; use local configuration")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None  # IPython magics are permitted; output removal still applies.
        if tree is not None:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                    continue
                value = node.value.value
                if not isinstance(value, str) or len(value) < 16:
                    continue
                names = [item.id.lower() for target in node.targets for item in ast.walk(target) if isinstance(item, ast.Name)]
                if any(word in name for name in names for word in keywords):
                    raise ValueError(f"Possible literal credential in code cell {index + 1}; use local configuration")
        cell["outputs"] = []
        cell["execution_count"] = None
        metadata = cell.setdefault("metadata", {})
        metadata.pop("execution", None)
        metadata.pop("ExecuteTime", None)
    notebook.setdefault("metadata", {}).pop("widgets", None)
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def main():
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig")
        encoded = clean_notebook(raw).encode("utf-8")
        sys.stdout.buffer.write(encoded)
    except Exception as exc:
        # Do not print input or credential values to the terminal.
        sys.stderr.write(f"notebook-clean: {type(exc).__name__}: {exc}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
