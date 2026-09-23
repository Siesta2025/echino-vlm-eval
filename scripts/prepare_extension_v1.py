"""Prepare the reviewed MedThinkVQA hard-negative extension (one image per case)."""

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/medthinkvqa"
DEST = ROOT / "data/extension-v1/images"
MANIFEST = ROOT / "data/manifests/extension-v1.jsonl"
AUDIT = ROOT / "data/extension-v1/audit.jsonl"
REVISION = "5555b80e167c4796efab92fe709fb05914077dc0"

# Frozen, manually screened source paths. These are new source cases, not extra
# views of any case in pilot.jsonl.
SELECTED = [
    "case00086/Wc0U6pAs.jpg",  # Caroli disease
    "case01394/jxo51PWH.jpg",  # Caroli disease
    "case01807/Hc4cmNTC.jpg",  # inflammatory pseudotumour
    "case01954/569fJo3Z.jpg",  # amoebic abscess
    "case02039/mk6yvSqc.jpg",  # mesenchymal hamartoma
    "case04624/6fwhQ8Ol.jpg",  # brucelloma
    "case11575/mepFhuai.jpg",  # mucinous cystadenoma
    "case13453/bQQv8riA.jpg",  # inflammatory pseudotumour
    "case15047/lr2VAaaF.jpg",  # amoebic abscess
    "case16653/va6_VQsM.jpg",  # peliosis
    "case17127/tOXSqRmo.jpg",  # endometriosis
    "case17560/dGP5GZmU.jpg",  # hemangiomatosis
    "case17675/n7cISUhv.jpg",  # necrotic nodule
    "case17786/rN9Bs5-L.jpg",  # fascioliasis
    "case18811/V99rtvmi.jpg",  # biliary hamartoma
    "case18953/8C1cLt8F.jpg",  # pyogenic abscess
]

EXCLUDED = {
    "case05099/tmF7UVaP.jpg": "visible_name_and_date_header_upload_hold",
    "case18031/A3JCwTEr.jpg": "two_view_composite_not_single_image",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_unchanged(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Existing output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    chosen = SELECTED + list(EXCLUDED)
    source_rows = {}
    for split in ("train", "test"):
        with (RAW / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                row = json.loads(line)
                for i in range(1, row["image_count"] + 1):
                    prefix = f"image_{i:02d}"
                    path = row.get(f"{prefix}_path")
                    if path and path.removeprefix("images/") in chosen:
                        source_rows[path.removeprefix("images/")] = (split, line_no, row, prefix)

    pilot_groups = {
        json.loads(line)["group_id"]
        for line in (ROOT / "data/manifests/pilot.jsonl").read_text(encoding="utf-8").splitlines()
    }
    manifest, audit = [], []
    for index, source_path in enumerate(chosen, 1):
        split, line_no, row, prefix = source_rows[source_path]
        case = source_path.split("/")[0]
        group_id = f"medthinkvqa-{case}"
        if group_id in pilot_groups:
            raise ValueError(f"Source case overlaps the pilot: {group_id}")
        raw_path = RAW / "images" / source_path
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if (row[f"{prefix}_modality"], row[f"{prefix}_sub_modality"]) != (
            "Ultrasound", "B-mode ultrasound"
        ):
            raise ValueError(f"Not a B-mode ultrasound image: {source_path}")
        with Image.open(raw_path) as image:
            fmt, dimensions = image.format, image.size
            image.verify()
        digest = sha256(raw_path)
        included = source_path in SELECTED
        case_id = f"extension-neg-{index:03d}"
        output_path = None
        if included:
            suffix = ".png" if fmt == "PNG" else ".jpg"
            output_path = DEST / f"neg{index:03d}{suffix}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                if sha256(output_path) != digest:
                    raise FileExistsError(f"Existing image differs: {output_path}")
            else:
                shutil.copy2(raw_path, output_path)
            manifest.append({
                "case_id": case_id,
                "group_id": group_id,
                "image_path": str(output_path.relative_to(ROOT)),
                "gold_label": "negative",
            })
        audit.append({
            "case_id": case_id,
            "group_id": group_id,
            "source_case": case.removeprefix("case"),
            "source_url": (
                "https://huggingface.co/datasets/bio-nlp-umass/MedThinkVQA/"
                f"resolve/{REVISION}/images/{source_path}"
            ),
            "metadata_revision": REVISION,
            "metadata_file": f"data/raw/medthinkvqa/{split}.jsonl",
            "metadata_line": line_no,
            "source_diagnosis": row["correct_answer_text"],
            "caption": row[f"{prefix}_caption"],
            "license": "CC-BY-NC-SA-4.0 (dataset card)",
            "modality": row[f"{prefix}_modality"],
            "sub_modality": row[f"{prefix}_sub_modality"],
            "raw_image_path": str(raw_path.relative_to(ROOT)),
            "image_path": str(output_path.relative_to(ROOT)) if output_path else None,
            "image_format": fmt,
            "dimensions": dimensions,
            "image_sha256": digest,
            "reference_label": "negative",
            "included": included,
            "reason": (
                "single_reviewed_b_mode_image_per_source_case"
                if included else EXCLUDED[source_path]
            ),
            "review": "visual_screen_and_source_evidence_checked_not_clinician_validated",
            "review_flags": ["educational_image_annotations_and_source_bias_require_reporting"],
        })

    manifest_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest)
    audit_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit)
    save_unchanged(MANIFEST, manifest_text)
    save_unchanged(AUDIT, audit_text)
    print(f"Prepared {len(manifest)} new negative cases; {len(EXCLUDED)} exclusions recorded")


if __name__ == "__main__":
    main()
