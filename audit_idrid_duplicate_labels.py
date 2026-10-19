"""
Audit duplicate IDRiD image pairs against official ground-truth annotations.

Audits six duplicate-image pairs discovered by hash-based image auditing:
- Group 1: IDRiD_118 (Training Set) vs IDRiD_064 (Testing Set)  [CROSS_SPLIT]
- Group 2: IDRiD_045 (Training Set) vs IDRiD_333 (Training Set)  [INTRA_TRAIN]
- Group 3: IDRiD_079 (Training Set) vs IDRiD_279 (Training Set)  [INTRA_TRAIN]
- Group 4: IDRiD_028 (Training Set) vs IDRiD_327 (Training Set)  [INTRA_TRAIN]
- Group 5: IDRiD_021 (Training Set) vs IDRiD_391 (Training Set)  [INTRA_TRAIN]
- Group 6: IDRiD_073 (Training Set) vs IDRiD_375 (Training Set)  [INTRA_TRAIN]

Official split annotation files:
- Training: B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv
- Testing:  B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv
"""

from pathlib import Path
import csv
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

# Project root: directory containing this script
ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT / "idrid_duplicate_label_audit.csv"

# Split-specific official annotation files
ANNOTATION_FILES: Dict[str, Path] = {
    "Training Set": ROOT / "B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
    "Testing Set": ROOT / "B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
}

# The six duplicate pairs to audit with their respective splits
DUPLICATE_PAIRS = [
    {
        "group": 1,
        "type": "CROSS_SPLIT",
        "item_a": {"id": "IDRiD_118", "split": "Training Set"},
        "item_b": {"id": "IDRiD_064", "split": "Testing Set"},
    },
    {
        "group": 2,
        "type": "INTRA_TRAIN",
        "item_a": {"id": "IDRiD_045", "split": "Training Set"},
        "item_b": {"id": "IDRiD_333", "split": "Training Set"},
    },
    {
        "group": 3,
        "type": "INTRA_TRAIN",
        "item_a": {"id": "IDRiD_079", "split": "Training Set"},
        "item_b": {"id": "IDRiD_279", "split": "Training Set"},
    },
    {
        "group": 4,
        "type": "INTRA_TRAIN",
        "item_a": {"id": "IDRiD_028", "split": "Training Set"},
        "item_b": {"id": "IDRiD_327", "split": "Training Set"},
    },
    {
        "group": 5,
        "type": "INTRA_TRAIN",
        "item_a": {"id": "IDRiD_021", "split": "Training Set"},
        "item_b": {"id": "IDRiD_391", "split": "Training Set"},
    },
    {
        "group": 6,
        "type": "INTRA_TRAIN",
        "item_a": {"id": "IDRiD_073", "split": "Training Set"},
        "item_b": {"id": "IDRiD_375", "split": "Training Set"},
    },
]

# Candidate header names (normalized) for ID and Grade columns
ID_COLUMNS = {
    "image", "image_id", "image name", "imagename",
    "id", "filename", "image_name"
}
GRADE_COLUMNS = {
    "grade", "dr_grade", "diagnosis", "disease_grade",
    "retinopathy grade", "retinopathy_grade",
    "diagnostic grade", "adjudicated grade"
}

# Strict pattern for IDRiD identifier normalization to prevent cross-ID collisions (e.g., IDRiD_021 vs IDRiD_210)
IMAGE_ID_PATTERN = re.compile(r"^IDRiD[_\s-]?0*(\d+)(?:\.[a-zA-Z0-9]+)?$", re.IGNORECASE)


def normalize_header(header: str) -> str:
    """Normalize a CSV header string for robust matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(header).lower()).strip()


def normalize_image_id(value: Any) -> Optional[str]:
    """
    Safely normalize IDRiD image identifiers to canonical 'IDRiD_XXX' format.
    Ensures exact token match without confusing e.g. IDRiD_021 with IDRiD_210.
    """
    match = IMAGE_ID_PATTERN.match(str(value).strip())
    if match:
        return f"IDRiD_{int(match.group(1)):03d}"
    return None


def load_split_annotations(csv_path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """
    Load an official annotation file safely without modifying it.
    Returns:
        records_by_id: dict mapping normalized image ID -> list of row records
        errors: list of error descriptions, if any
    """
    records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    errors: List[str] = []

    if not csv_path.exists():
        errors.append(f"Annotation file not found: {csv_path}")
        return records, errors

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                errors.append(f"Annotation file is empty or missing headers: {csv_path}")
                return records, errors

            # Map normalized column names to original headers
            header_map = {
                normalize_header(header): header
                for header in reader.fieldnames
                if header is not None and header.strip()
            }

            id_headers = [orig for norm, orig in header_map.items() if norm in ID_COLUMNS]
            grade_headers = [orig for norm, orig in header_map.items() if norm in GRADE_COLUMNS]

            if len(id_headers) != 1:
                errors.append(
                    f"Ambiguous or missing ID column in {csv_path.name}: found {id_headers}"
                )
                return records, errors

            if len(grade_headers) != 1:
                errors.append(
                    f"Ambiguous or missing Grade column in {csv_path.name}: found {grade_headers}"
                )
                return records, errors

            id_col = id_headers[0]
            grade_col = grade_headers[0]

            for row_num, row in enumerate(reader, start=2):
                raw_id = row.get(id_col)
                if raw_id is None or not str(raw_id).strip():
                    continue

                norm_id = normalize_image_id(raw_id)
                if not norm_id:
                    continue

                raw_grade = row.get(grade_col)
                grade = str(raw_grade).strip() if raw_grade is not None else ""

                try:
                    rel_source = str(csv_path.relative_to(ROOT))
                except ValueError:
                    rel_source = str(csv_path)

                records[norm_id].append({
                    "raw_id": raw_id,
                    "grade": grade,
                    "row": row_num,
                    "source": rel_source,
                })

    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"Failed to read {csv_path}: {exc}")

    return records, errors


def resolve_image_record(
    records: List[Dict[str, Any]], expected_split_source: str
) -> Tuple[str, str, str]:
    """
    Resolve the official grade and provenance for a single image within its split.
    Returns:
        (official_grade, annotation_source, status)
    """
    if not records:
        return "", "NO_MATCHING_ANNOTATION_ROW", "UNVERIFIED"

    non_empty_grades = {r["grade"] for r in records if r["grade"] != ""}

    if not non_empty_grades:
        return "", records[0]["source"], "MISSING_GRADE"

    if len(non_empty_grades) > 1:
        return "", "CONFLICTING_ANNOTATION_RECORDS", "CONFLICT"

    # Exactly one non-empty grade found
    grade = next(iter(non_empty_grades))
    source = records[0]["source"]
    return grade, source, "RESOLVED"


def main():
    print(f"Project root: {ROOT}")
    print("Loading official split annotation files...")

    annotations_by_split: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    load_errors: List[str] = []

    for split_name, csv_path in ANNOTATION_FILES.items():
        records, errs = load_split_annotations(csv_path)
        annotations_by_split[split_name] = records
        load_errors.extend(errs)
        try:
            rel_path = csv_path.relative_to(ROOT)
        except ValueError:
            rel_path = csv_path
        print(f"  • {split_name}: loaded {len(records)} image records from '{rel_path}'")

    if load_errors:
        print("\nWarnings/Errors loading annotation files:")
        for err in load_errors:
            print(f"  - {err}")

    output_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print("DUPLICATE PAIR AUDIT SUMMARY")
    print("=" * 90)

    for pair in DUPLICATE_PAIRS:
        group_num = pair["group"]
        pair_type = pair["type"]

        item_a = pair["item_a"]
        item_b = pair["item_b"]

        id_a, split_a = item_a["id"], item_a["split"]
        id_b, split_b = item_b["id"], item_b["split"]

        records_a = annotations_by_split.get(split_a, {}).get(id_a, [])
        records_b = annotations_by_split.get(split_b, {}).get(id_b, [])

        grade_a, source_a, status_a = resolve_image_record(
            records_a, str(ANNOTATION_FILES.get(split_a, ""))
        )
        grade_b, source_b, status_b = resolve_image_record(
            records_b, str(ANNOTATION_FILES.get(split_b, ""))
        )

        # Determine pair_label_match
        if status_a != "RESOLVED" or status_b != "RESOLVED" or not grade_a or not grade_b:
            pair_label_match = "UNVERIFIED"
        elif grade_a == grade_b:
            pair_label_match = "MATCH"
        else:
            pair_label_match = "LABEL_MISMATCH"

        # Determine review_status
        if pair_type == "CROSS_SPLIT":
            review_status = "REVIEW_CROSS_SPLIT_LEAKAGE"
        elif pair_label_match == "LABEL_MISMATCH":
            review_status = "MANUAL_REVIEW_REQUIRED"
        elif pair_label_match == "MATCH":
            review_status = "LABELS_MATCH_CHECK_PROVENANCE"
        else:
            review_status = "UNVERIFIED_DATASET_RECORD"

        output_rows.append({
            "duplicate_group": group_num,
            "image_id": id_a,
            "split": split_a,
            "official_grade": grade_a,
            "annotation_source": source_a,
            "pair_label_match": pair_label_match,
            "review_status": review_status,
        })
        output_rows.append({
            "duplicate_group": group_num,
            "image_id": id_b,
            "split": split_b,
            "official_grade": grade_b,
            "annotation_source": source_b,
            "pair_label_match": pair_label_match,
            "review_status": review_status,
        })

        grade_disp_a = f"Grade {grade_a}" if grade_a else f"[{status_a}]"
        grade_disp_b = f"Grade {grade_b}" if grade_b else f"[{status_b}]"

        print(
            f"Group {group_num} ({pair_type}):\n"
            f"  Image A: {id_a} ({split_a}) -> {grade_disp_a} | Source: {source_a}\n"
            f"  Image B: {id_b} ({split_b}) -> {grade_disp_b} | Source: {source_b}\n"
            f"  Result : {pair_label_match} | Review Action: {review_status}\n"
        )

    fieldnames = [
        "duplicate_group",
        "image_id",
        "split",
        "official_grade",
        "annotation_source",
        "pair_label_match",
        "review_status",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("=" * 90)
    print("Audit complete.")
    print(f"Report saved to: {OUTPUT_FILE}")
    print("\nPair Status Definitions:")
    print("  MATCH          : Official grades in respective annotation files agree.")
    print("  LABEL_MISMATCH : Official grades in respective annotation files differ.")
    print("  UNVERIFIED     : Missing row, empty grade, or conflicting records within the split file.")
    print("\nSafety Notice:")
    print("  No images, annotations, models, or configs were modified.")
    print("  Conflicting grades represent dataset annotation ambiguities requiring")
    print("  verification against original clinical study documentation.")


if __name__ == "__main__":
    main()
