#!/usr/bin/env python3
"""Check that a documentation change touches only comments and docstrings.

Usage:
    python docs/tools/check_comments_only.py BASE [HEAD]

BASE and HEAD are git revisions (HEAD defaults to the working tree's HEAD).
For every source file that exists in both revisions and differs, the code is
compared with comments and docstrings removed:

- Python (.py): the syntax trees are compared after removing docstrings.
- C and C++ (.h, .hh, .hpp, .c, .cc, .cpp, .cu, .cuh): the text is compared
  after removing // and /* */ comments and all whitespace. String and
  character literals are kept.

Source files that were added, deleted or renamed are reported as code changes.
Files under docs/ (the tooling of the documentation site itself) and files of
other types (Markdown, YAML, ...) are not checked. The exit status is 1 if any
source file has a code change, 0 otherwise.
"""
import ast
import re
import subprocess
import sys

PY = (".py",)
C_LIKE = (".h", ".hh", ".hpp", ".c", ".cc", ".cpp", ".cu", ".cuh")

# A comment, or a string/character literal that has to be skipped as a whole so
# that "//" or "/*" inside it is not taken for a comment.
_C_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def show(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], check=True,
                          capture_output=True).stdout.decode("utf-8", "replace")


def python_code(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            rest = body[1:]
            node.body = rest if rest or isinstance(node, ast.Module) else [ast.Pass()]
    return ast.dump(tree)


def c_code(src):
    stripped = _C_TOKEN.sub(lambda m: " " if m.group(0).startswith("/") else m.group(0), src)
    return "".join(stripped.split())


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    base = argv[1]
    head = argv[2] if len(argv) == 3 else "HEAD"
    changes = git("diff", "--name-status", "--no-renames", f"{base}...{head}").splitlines()
    problems, checked = [], 0
    for line in changes:
        status, path = line.split("\t", 1)
        if path.startswith("docs/") or not path.endswith(PY + C_LIKE):
            continue
        if status != "M":
            problems.append(f"{path}: source file {'added' if status == 'A' else 'deleted'}")
            continue
        old, new = show(base, path), show(head, path)
        checked += 1
        try:
            same = (python_code(old) == python_code(new)) if path.endswith(PY) else (c_code(old) == c_code(new))
        except SyntaxError as err:
            problems.append(f"{path}: cannot parse ({err.msg}); check by hand")
            continue
        if not same:
            problems.append(f"{path}: code changed (not only comments or docstrings)")
    print(f"checked {checked} modified source files")
    for p in problems:
        print("  " + p)
    if problems:
        print("FAIL: this change touches more than comments and docstrings")
        return 1
    print("OK: only comments and docstrings changed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
