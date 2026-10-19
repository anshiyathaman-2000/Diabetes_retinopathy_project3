
from pathlib import Path
import hashlib
import csv
from collections import defaultdict

# IDRiD image folders
TRAIN_DIR = Path("B. Disease Grading/1. Original Images/a. Training Set")
TEST_DIR = Path("B. Disease Grading/1. Original Images/b. Testing Set")

OUTPUT_FILE = Path("idrid_duplicate_audit.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    hash_groups = defaultdict(list)

    for folder in (TRAIN_DIR, TEST_DIR):
        if not folder.is_dir():
            print(f"WARNING: Folder not found: {folder}")
            continue

        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                digest = sha256_file(path)
                hash_groups[digest].append(str(path))

    duplicates = {
        digest: paths
        for digest, paths in hash_groups.items()
        if len(paths) > 1
    }

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sha256", "number_of_copies", "file_path"])

        for digest, paths in duplicates.items():
            for path in paths:
                writer.writerow([digest, len(paths), path])

    print(f"Images scanned: {sum(map(len, hash_groups.values()))}")
    print(f"Duplicate groups found: {len(duplicates)}")
    print(f"Audit report saved to: {OUTPUT_FILE.resolve()}")

    if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        print("IMPORTANT: Check and correct the image folder paths above.")


if __name__ == "__main__":
    main()
