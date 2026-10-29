import re
import sys
from pathlib import Path

def strip_fstring_prefixes(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    # Matches any f-string: f'...', f"...", even in concatenations
    fstring_pattern = re.compile(r"""(?<!\w)f(['"])(.*?)(\1)""")

    def replace_fstring(match):
        quote = match.group(1)
        content = match.group(2)
        return f'{quote}{content}{quote}'

    new_lines = []
    for line in lines:
        if 'f"' in line or "f'" in line:
            newline = fstring_pattern.sub(replace_fstring, line)
            new_lines.append(newline)
        else:
            new_lines.append(line)

    backup_path = Path(file_path).with_suffix('.bak')
    Path(file_path).rename(backup_path)
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

    print(f"Updated {file_path}. Original backed up as {backup_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_fstrings_anywhere.py rix.py")
        sys.exit(1)

    strip_fstring_prefixes(sys.argv[1])
