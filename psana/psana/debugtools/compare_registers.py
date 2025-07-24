import yaml
import sys

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def flatten_dict(d, parent_key='', sep='.'):
    """Flattens nested dictionaries into a single-level dict with dotted paths."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def compare_registers(file1, file2):
    data1 = flatten_dict(load_yaml(file1))
    data2 = flatten_dict(load_yaml(file2))

    all_keys = set(data1.keys()).union(data2.keys())

    print(f"\nDifferences between:\n  {file1}\n  {file2}\n")
    for key in sorted(all_keys):
        val1 = data1.get(key, "<missing>")
        val2 = data2.get(key, "<missing>")
        if val1 != val2:
            print(f"{key}:\n  {file1}: {val1}\n  {file2}: {val2}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_registers.py <file1.yaml> <file2.yaml>")
        sys.exit(1)

    compare_registers(sys.argv[1], sys.argv[2])
