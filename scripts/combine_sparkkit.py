import os
import re
import sys


def combine_py_files(src_dir, output_file):
    """
    Combines all .py files in the specified source directory (excluding __init__.py)
    into a single Python file with imports consolidated at the top.

    Args:
        src_dir (str): Path to the source directory containing .py files.
        output_file (str): Path to the output file to be created.
    """
    import_lines = []
    code_lines = []
    import_pattern = re.compile(r"^(import |from )")

    # List all .py files excluding __init__.py
    py_files = sorted(
        [f for f in os.listdir(src_dir) if f.endswith(".py") and f != "__init__.py"]
    )

    for py_file in py_files:
        file_path = os.path.join(src_dir, py_file)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        file_imports = []
        file_code = []
        for line in lines:
            if import_pattern.match(line):
                # Exclude internal imports (relative or within sparkkit)
                if not re.match(r"^from\s+\.|^from\s+sparkkit\.", line):
                    file_imports.append(line)
                # Optionally, handle or modify internal imports here
            else:
                file_code.append(line)

        import_lines.extend(file_imports)
        # Add a comment to indicate the source file for clarity
        code_lines.append(f"# Contents of {py_file}\n")
        code_lines.extend(file_code)
        code_lines.append("\n")  # Ensure separation between files

    # Deduplicate import lines while preserving order
    seen = set()
    dedup_imports = []
    for line in import_lines:
        if line not in seen:
            dedup_imports.append(line)
            seen.add(line)

    # Combine all parts: imports first, then the combined code
    combined_content = dedup_imports + ["\n"] + code_lines

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the combined content to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(combined_content)

    print(f"Combined file created at '{output_file}'.")


if __name__ == "__main__":
    # Define the source directory and output file paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_directory = os.path.join(project_root, "src", "sparkkit")
    output_directory = os.path.join(project_root, "combined")
    output_filename = "sparkkit_combined.py"
    output_filepath = os.path.join(output_directory, output_filename)

    # Ensure the source directory exists
    if not os.path.isdir(src_directory):
        print(f"Source directory '{src_directory}' does not exist.", file=sys.stderr)
        sys.exit(1)
    else:
        combine_py_files(src_directory, output_filepath)
