import os
import re
import json
from collections import defaultdict

# === Configuration ===
directory = "../"
excluded_dirs = {"Old_files","BetulFuzzyMatch","rebekah-idea-testing"}

project_root = os.path.abspath(directory)

# === Regex Patterns ===
assign_literal_pattern = re.compile(r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]")
assign_list_pattern = re.compile(r"(\w+)\s*=\s*\[([^\]]+)\]")
string_item_pattern = re.compile(r"['\"]([^'\"]+)['\"]")
loop_over_var_pattern = re.compile(r"for\s+(\w+)\s+in\s+(\w+)")
re_match_pattern = re.compile(r"re\.match\((r?['\"].*?['\"]),\s*(\w+)\)")

# Updated pandas I/O patterns to capture only first string argument, ignoring trailing kwargs
read_patterns = [
    r"pd\.read_csv\(\s*(['\"][^'\"]+['\"])",
    r"pd\.read_excel\(\s*(['\"][^'\"]+['\"])",
    r"pd\.read_json\(\s*(['\"][^'\"]+['\"])",
    r"pd\.read_parquet\(\s*(['\"][^'\"]+['\"])",
    r"pd\.read_feather\(\s*(['\"][^'\"]+['\"])",
    r"pd\.read_pickle\(\s*(['\"][^'\"]+['\"])",
]

write_patterns = [
    r"\.to_csv\(\s*(['\"][^'\"]+['\"])",
    r"\.to_excel\(\s*(['\"][^'\"]+['\"])",
    r"\.to_json\(\s*(['\"][^'\"]+['\"])",
    r"\.to_parquet\(\s*(['\"][^'\"]+['\"])",
    r"\.to_feather\(\s*(['\"][^'\"]+['\"])",
    r"\.to_pickle\(\s*(['\"][^'\"]+['\"])",
]

read_regexes = [re.compile(p) for p in read_patterns]
write_regexes = [re.compile(p) for p in write_patterns]

# === File Tracking ===
file_reads = defaultdict(set)
file_writes = defaultdict(set)

def extract_list_items(list_str):
    return string_item_pattern.findall(list_str)

def resolve_arg(arg, var_map, current_file_dir, project_root):
    arg = arg.strip()
    resolved = []

    # Remove enclosing quotes if present
    if arg.startswith('"') or arg.startswith("'"):
        raw_path = arg.strip('"\'')
        full_path = os.path.normpath(os.path.join(current_file_dir, raw_path))
        resolved.append(os.path.relpath(full_path, project_root))
    elif arg in var_map:
        for val in var_map[arg]:
            full_path = os.path.normpath(os.path.join(current_file_dir, val))
            resolved.append(os.path.relpath(full_path, project_root))

    return resolved

def process_lines(lines, source_file, current_file_dir, project_root):
    var_map = {}
    loop_context = {}

    for line in lines:
        stripped = line.strip()

        if m := assign_literal_pattern.match(stripped):
            var_map[m.group(1)] = [m.group(2)]
        elif m := assign_list_pattern.match(stripped):
            var_map[m.group(1)] = extract_list_items(m.group(2))

        if m := loop_over_var_pattern.match(stripped):
            loop_var, list_var = m.group(1), m.group(2)
            loop_context[loop_var] = var_map.get(list_var, [])
            continue

        for regex in read_regexes:
            if m := regex.search(stripped):
                arg = m.group(1)
                filenames = resolve_arg(arg, var_map, current_file_dir, project_root)
                if not filenames and arg in loop_context:
                    filenames = loop_context[arg]
                for f in filenames:
                    file_reads[f].add(source_file)

        for regex in write_regexes:
            if m := regex.search(stripped):
                arg = m.group(1)
                filenames = resolve_arg(arg, var_map, current_file_dir, project_root)
                if not filenames and arg in loop_context:
                    filenames = loop_context[arg]
                for f in filenames:
                    file_writes[f].add(source_file)

        if m := re_match_pattern.search(stripped):
            pattern = m.group(1).strip('"\'')
            file_reads[f"(dynamic match: {pattern})"].add(source_file)

def process_file(path):
    try:
        current_file_dir = os.path.dirname(path)
        if path.endswith(".py"):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            process_lines(lines, path, current_file_dir, project_root)
        elif path.endswith(".ipynb"):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                nb = json.load(f)
            lines = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    lines.extend(cell.get("source", []))
            process_lines(lines, path, current_file_dir, project_root)
    except Exception as e:
        print(f"Error processing {path}: {e}")

# === Walk directory ===
for root, dirs, files in os.walk(directory):
    dirs[:] = [d for d in dirs if d not in excluded_dirs]
    for fname in files:
        if fname.endswith((".py", ".ipynb")):
            process_file(os.path.join(root, fname))

# === Output Summary ===
print("=== Data File Access Summary ===")
all_data_files = sorted(set(file_reads) | set(file_writes))
for data_file in all_data_files:
    print(f"\n{data_file}")
    if readers := file_reads.get(data_file):
        print("  Read by:")
        for src in sorted(readers):
            print(f"    {src}")
    if writers := file_writes.get(data_file):
        print("  Written by:")
        for src in sorted(writers):
            print(f"    {src}")
