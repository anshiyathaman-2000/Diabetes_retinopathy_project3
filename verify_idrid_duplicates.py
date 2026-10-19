"""
Independent reproducibility verification script for IDRiD duplicate images.

Verifies:
1. Existence and locations of all duplicate images in Training and Testing splits.
2. SHA-256 hash calculation via hashlib for all images.
3. Byte-identical status comparison.
4. Decoded pixel array comparison (via PIL and numpy) for pixel-identical status.
5. Official ground-truth grade retrieval from split-specific CSV annotation files.
6. Export of verification report to 'idrid_duplicate_verification_report.csv'.
"""

from pathlib import Path
import hashlib
import csv
import re
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT / "idrid_duplicate_verification_report.csv"

# Split directory mapping
SPLIT_DIRS: Dict[str, Path] = {
    "Training Set": ROOT / "B. Disease Grading/1. Original Images/a. Training Set",
    "Testing Set": ROOT / "B. Disease Grading/1. Original Images/b. Testing Set",
}

# Split official annotation file mapping
ANNOTATION_FILES: Dict[str, Path] = {
    "Training Set": ROOT / "B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
    "Testing Set": ROOT / "B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
}

# The six duplicate pairs to verify
AUDIT_PAIRS = [
    {
        "group": 1,
        "type": "CROSS_SPLIT",
        "a": {"id": "IDRiD_118", "split": "Training Set"},
        "b": {"id": "IDRiD_064", "split": "Testing Set"},
    },
    {
        "group": 2,
        "type": "INTRA_TRAIN",
        "a": {"id": "IDRiD_045", "split": "Training Set"},
        "b": {"id": "IDRiD_333", "split": "Training Set"},
    },
    {
        "group": 3,
        "type": "INTRA_TRAIN",
        "a": {"id": "IDRiD_079", "split": "Training Set"},
        "b": {"id": "IDRiD_279", "split": "Training Set"},
    },
    {
        "group": 4,
        "type": "INTRA_TRAIN",
        "a": {"id": "IDRiD_028", "split": "Training Set"},
        "b": {"id": "IDRiD_327", "split": "Training Set"},
    },
    {
        "group": 5,
        "type": "INTRA_TRAIN",
        "a": {"id": "IDRiD_021", "split": "Training Set"},
        "b": {"id": "IDRiD_391", "split": "Training Set"},
    },
    {
        "group": 6,
        "type": "INTRA_TRAIN",
        "a": {"id": "IDRiD_073", "split": "Training Set"},
        "b": {"id": "IDRiD_375", "split": "Training Set"},
    },
]

IMAGE_ID_PATTERN = re.compile(r"^IDRiD[_\s-]?0*(\d+)(?:\.[a-zA-Z0-9]+)?$", re.IGNORECASE)


def normalize_image_id(value: Any) -> Optional[str]:
    """Normalize IDRiD image ID to standard canonical form (e.g. IDRiD_021)."""
    match = IMAGE_ID_PATTERN.match(str(value).strip())
    if match:
        return f"IDRiD_{int(match.group(1)):03d}"
    return None


def calculate_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file reading in 1MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_grades(annotation_file: Path) -> Dict[str, str]:
    """Load official disease grades from a split annotation CSV."""
    grades: Dict[str, str] = {}
    if not annotation_file.exists():
        return grades

    with annotation_file.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Find ID and Grade columns
        id_col = None
        grade_col = None
        for col in reader.fieldnames or []:
            norm_col = re.sub(r"[^a-z0-9]+", " ", col.lower()).strip()
            if norm_col in {"image", "image_id", "image name", "imagename", "id", "filename"}:
                id_col = col
            if norm_col in {"grade", "dr_grade", "retinopathy grade", "retinopathy_grade", "disease_grade"}:
                grade_col = col

        if not id_col or not grade_col:
            return grades

        for row in reader:
            raw_id = row.get(id_col)
            norm_id = normalize_image_id(raw_id)
            if norm_id:
                raw_grade = row.get(grade_col)
                grades[norm_id] = str(raw_grade).strip() if raw_grade is not None else ""
    return grades


def find_image_file(folder: Path, image_id: str) -> Optional[Path]:
    """Find the image file corresponding to an ID (e.g. IDRiD_021.jpg)."""
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG"]:
        candidate = folder / f"{image_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def verify_pair(
    pair: Dict[str, Any],
    train_grades: Dict[str, str],
    test_grades: Dict[str, str]
) -> Dict[str, Any]:
    """Perform independent hash, dimension, pixel array, and grade verification."""
    group_num = pair["group"]
    pair_type = pair["type"]

    item_a = pair["a"]
    item_b = pair["b"]

    id_a, split_a = item_a["id"], item_a["split"]
    id_b, split_b = item_b["id"], item_b["split"]

    dir_a = SPLIT_DIRS.get(split_a)
    dir_b = SPLIT_DIRS.get(split_b)

    file_a = find_image_file(dir_a, id_a) if dir_a else None
    file_b = find_image_file(dir_b, id_b) if dir_b else None

    # Get official grades
    grade_a = train_grades.get(id_a, "") if split_a == "Training Set" else test_grades.get(id_a, "")
    grade_b = train_grades.get(id_b, "") if split_b == "Training Set" else test_grades.get(id_b, "")

    if not file_a or not file_b:
        return {
            "group": group_num,
            "pair_type": pair_type,
            "image_a": id_a,
            "split_a": split_a,
            "grade_a": grade_a,
            "sha256_a": calculate_sha256(file_a) if file_a else "",
            "image_b": id_b,
            "split_b": split_b,
            "grade_b": grade_b,
            "sha256_b": calculate_sha256(file_b) if file_b else "",
            "byte_identical": False,
            "pixel_identical": False,
            "dim_a": None,
            "dim_b": None,
            "size_a": file_a.stat().st_size if file_a else 0,
            "size_b": file_b.stat().st_size if file_b else 0,
            "verification_status": "UNVERIFIED",
            "notes": f"Missing file: {'A' if not file_a else ''} {'B' if not file_b else ''}".strip(),
        }

    try:
        sha256_a = calculate_sha256(file_a)
        sha256_b = calculate_sha256(file_b)
        byte_identical = (sha256_a == sha256_b)

        # Dimension and pixel verification via PIL and numpy
        with Image.open(file_a) as im_a, Image.open(file_b) as im_b:
            dim_a = (im_a.width, im_a.height)
            dim_b = (im_b.width, im_b.height)
            mode_a = im_a.mode
            mode_b = im_b.mode

            arr_a = np.array(im_a)
            arr_b = np.array(im_b)
            pixel_identical = bool(np.array_equal(arr_a, arr_b))

        verification_status = "VERIFIED"
        notes = "OK"

    except Exception as exc:
        return {
            "group": group_num,
            "pair_type": pair_type,
            "image_a": id_a,
            "split_a": split_a,
            "grade_a": grade_a,
            "sha256_a": "",
            "image_b": id_b,
            "split_b": split_b,
            "grade_b": grade_b,
            "sha256_b": "",
            "byte_identical": False,
            "pixel_identical": False,
            "dim_a": None,
            "dim_b": None,
            "size_a": 0,
            "size_b": 0,
            "verification_status": "UNVERIFIED",
            "notes": f"Read error: {exc}",
        }

    return {
        "group": group_num,
        "pair_type": pair_type,
        "image_a": id_a,
        "split_a": split_a,
        "grade_a": grade_a,
        "sha256_a": sha256_a,
        "image_b": id_b,
        "split_b": split_b,
        "grade_b": grade_b,
        "sha256_b": sha256_b,
        "byte_identical": byte_identical,
        "pixel_identical": pixel_identical,
        "dim_a": f"{dim_a[0]}x{dim_a[1]}",
        "dim_b": f"{dim_b[0]}x{dim_b[1]}",
        "mode_a": mode_a,
        "mode_b": mode_b,
        "size_a": file_a.stat().st_size,
        "size_b": file_b.stat().st_size,
        "verification_status": verification_status,
        "notes": notes,
    }


def main():
    print(f"Project root: {ROOT}")
    print("Loading official split ground-truth annotations...")
    train_grades = load_official_grades(ANNOTATION_FILES["Training Set"])
    test_grades = load_official_grades(ANNOTATION_FILES["Testing Set"])
    print(f"  • Training grades loaded: {len(train_grades)}")
    print(f"  • Testing grades loaded : {len(test_grades)}")

    print("\nAuditing image files and hashes for all 6 pairs...")
    results: List[Dict[str, Any]] = []

    for pair in AUDIT_PAIRS:
        res = verify_pair(pair, train_grades, test_grades)
        results.append(res)

    # Print summary
    print("\n" + "=" * 105)
    print(f"{'Grp':<4} {'Image A':<11} {'Split A':<14} {'Grd A':<6} {'Image B':<11} {'Split B':<14} {'Grd B':<6} {'Byte-Identical':<16} {'Pixel-Identical':<17} {'Status'}")
    print("=" * 105)

    for r in results:
        byte_str = "YES" if r["byte_identical"] else "NO"
        pixel_str = "YES" if r["pixel_identical"] else "NO"
        print(
            f"{r['group']:<4} "
            f"{r['image_a']:<11} "
            f"{r['split_a']:<14} "
            f"{r['grade_a']:<6} "
            f"{r['image_b']:<11} "
            f"{r['split_b']:<14} "
            f"{r['grade_b']:<6} "
            f"{byte_str:<16} "
            f"{pixel_str:<17} "
            f"{r['verification_status']}"
        )

    print("=" * 105)
    print("\nDetailed Pair-by-Pair Evidence:")
    for r in results:
        print(f"\n[Group {r['group']}] ({r['pair_type']})")
        print(f"  • Image A: {r['image_a']} ({r['split_a']}) | Grade: {r['grade_a']} | Size: {r['size_a']:,} bytes | Dims: {r['dim_a']}")
        print(f"    SHA-256: {r['sha256_a']}")
        print(f"  • Image B: {r['image_b']} ({r['split_b']}) | Grade: {r['grade_b']} | Size: {r['size_b']:,} bytes | Dims: {r['dim_b']}")
        print(f"    SHA-256: {r['sha256_b']}")
        print(f"  • Byte-Identical : {r['byte_identical']}")
        print(f"  • Pixel-Identical: {r['pixel_identical']}")
        print(f"  • Grade Status   : {'MATCH' if r['grade_a'] == r['grade_b'] else 'LABEL_MISMATCH'}")
        print(f"  • Audit Status   : {r['verification_status']}")

    # Export CSV
    fieldnames = [
        "group",
        "image_a",
        "split_a",
        "grade_a",
        "sha256_a",
        "image_b",
        "split_b",
        "grade_b",
        "sha256_b",
        "byte_identical",
        "pixel_identical",
        "verification_status",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "group": r["group"],
                "image_a": r["image_a"],
                "split_a": r["split_a"],
                "grade_a": r["grade_a"],
                "sha256_a": r["sha256_a"],
                "image_b": r["image_b"],
                "split_b": r["split_b"],
                "grade_b": r["grade_b"],
                "sha256_b": r["sha256_b"],
                "byte_identical": r["byte_identical"],
                "pixel_identical": r["pixel_identical"],
                "verification_status": r["verification_status"],
            })

    print(f"\nVerification report generated at: {OUTPUT_CSV}")
    unresolved = [r for r in results if r["verification_status"] == "UNVERIFIED"]
    if unresolved:
        print(f"WARNING: {len(unresolved)} pairs were UNVERIFIED.")
    else:
        print("All 6 pairs were successfully resolved and verified.")


if __name__ == "__main__":
    main()
