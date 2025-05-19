import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT_NAME = "regime_predictor_lib"
OUTPUT_FILENAME = "project_snapshot.txt"
LOG_FILENAME = "snapshot_generator.log"
SEPARATOR = "\n--------------\n"


IGNORE_DIRS = {
    ".venv",
    "venv",
    "env",
    "ENV",
    ".git",
    ".dvc",
    "__pycache__",
    ".idea",
    ".vscode",
}
IGNORE_FILES = {
    ".DS_Store",
    "Thumbs.db",
    OUTPUT_FILENAME,
    LOG_FILENAME,
}
IGNORE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".egg-info",
    ".sqlite3-journal",
    ".db-journal",
}

TEXT_FILE_EXTENSIONS = {
    ".py",
    ".sql",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".json",
    ".sh",
    ".bash",
    ".gitignore",
    ".dockerignore",
    ".env",
    ".example",
    ".csv",
    ".html",
    ".css",
    ".js",
}


def should_ignore(path_item: Path, project_root: Path) -> bool:
    try:
        relative_to_root = path_item.relative_to(project_root)
        if relative_to_root.parts and relative_to_root.parts[0] in IGNORE_DIRS:
            return True
    except ValueError:
        pass

    if path_item.name in IGNORE_DIRS or path_item.name in IGNORE_FILES:
        return True
    if path_item.is_file() and path_item.suffix in IGNORE_EXTENSIONS:
        return True
    return False


def is_text_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in TEXT_FILE_EXTENSIONS


def scan_project(
    start_path: Path, project_root: Path, output_file, logger_instance: logging.Logger
):
    for item in sorted(start_path.iterdir()):
        if should_ignore(item, project_root):
            logger_instance.info(f"Ignoring: {item.relative_to(project_root)}")
            continue

        relative_path_str = str(item.relative_to(project_root))
        output_file.write(f"PATH: {relative_path_str}\n")

        if item.is_dir():
            output_file.write("TYPE: DIRECTORY\n")
            is_empty = True
            for sub_item in item.iterdir():
                if not should_ignore(sub_item, project_root):
                    is_empty = False
                    break
            if is_empty:
                output_file.write("STATUS: EMPTY\n")

            output_file.write(SEPARATOR)
            scan_project(item, project_root, output_file, logger_instance)

        elif item.is_file():
            output_file.write("TYPE: FILE\n")
            if is_text_file(item):
                output_file.write("CONTENT:\n")
                try:
                    with open(
                        item, "r", encoding="utf-8", errors="surrogateescape"
                    ) as f_content:
                        content = f_content.read()
                        output_file.write(content)
                except Exception as e:
                    output_file.write(f"[Error reading file: {e}]\n")
            else:
                output_file.write(
                    f"[Binary or non-text file (extension: {item.suffix}) "
                    "- content skipped]\n"
                )
            output_file.write(SEPARATOR)
        else:
            output_file.write("TYPE: OTHER (e.g., symlink) - SKIPPED\n")
            output_file.write(SEPARATOR)


if __name__ == "__main__":
    current_script_path = Path(__file__).resolve()
    project_root_final = current_script_path.parent

    found_root_override = None
    path_check = current_script_path
    for _ in range(5):
        if path_check.is_dir() and path_check.name == PROJECT_ROOT_NAME:
            found_root_override = path_check
            break
        if path_check.parent == path_check:
            break
        path_check = path_check.parent

    root_determination_message = ""
    is_warning_message = False

    if found_root_override:
        project_root_final = found_root_override
        root_determination_message = f"Determined project root: {project_root_final}"
    else:
        root_determination_message = (
            f"Could not find directory named '{PROJECT_ROOT_NAME}' "
            "at or above script location. "
            "Using script's parent directory as "
            f"project root: {project_root_final}"
        )
        is_warning_message = True

    log_file_path = project_root_final / LOG_FILENAME

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    if is_warning_message:
        logger.warning(root_determination_message)
    else:
        logger.info(root_determination_message)

    output_path = project_root_final / OUTPUT_FILENAME

    logger.info(f"Scanning project at: {project_root_final}")
    logger.info(f"Output will be saved to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as outfile:
        scan_project(project_root_final, project_root_final, outfile, logger)

    logger.info(f"Project snapshot created: {output_path}")
