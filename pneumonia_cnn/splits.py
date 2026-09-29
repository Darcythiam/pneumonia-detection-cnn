"""Dataset discovery and stratified train/validation splitting."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from sklearn.model_selection import train_test_split


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
EXTENSIONS = {".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int


def resolve_data_root(path: str | Path) -> Path:
    """Accept a direct train/val/test root or the common extra chest_xray wrapper."""
    base = Path(path).expanduser().resolve()
    for candidate in (base, base / "chest_xray"):
        if all((candidate / split / name).is_dir()
               for split in ("train", "test") for name in CLASS_NAMES):
            return candidate
    raise FileNotFoundError(
        f"Expected train/NORMAL, train/PNEUMONIA, test/NORMAL, and "
        f"test/PNEUMONIA under {base} (or its chest_xray child)."
    )


def collect_split(root: Path, split: str) -> list[ImageRecord]:
    records = []
    for label, name in enumerate(CLASS_NAMES):
        folder = root / split / name
        if not folder.is_dir():
            if split == "val":
                continue  # Some copies of the dataset omit the tiny original val split.
            raise FileNotFoundError(folder)
        records.extend(ImageRecord(path, label) for path in sorted(folder.rglob("*"))
                       if path.is_file() and path.suffix.lower() in EXTENSIONS)
    return sorted(records, key=lambda record: str(record.path))


def image_signature(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as image:
        for chunk in iter(lambda: image.read(1024 * 1024), b""):
            digest.update(chunk)
    return path.stat().st_size, digest.hexdigest()


def deduplicate_pool(records: list[ImageRecord],
                     removed: list[tuple[Path, Path]] | None = None) -> list[ImageRecord]:
    """Keep the first source image for each exact byte sequence before splitting."""
    seen: dict[tuple[int, str], ImageRecord] = {}
    unique = []
    for record in records:
        signature = image_signature(record.path)
        earlier = seen.get(signature)
        if earlier is None:
            seen[signature] = record
            unique.append(record)
        elif earlier.label != record.label:
            raise ValueError(f"Identical images with conflicting labels: "
                             f"{earlier.path}, {record.path}")
        elif removed is not None:
            removed.append((record.path, earlier.path))
    return unique


def make_splits(data_root: str | Path, *, validation_fraction: float = 0.2,
                seed: int = 42, check_duplicates: bool = True,
                duplicate_report: list[tuple[Path, Path]] | None = None
                ) -> dict[str, list[ImageRecord]]:
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5")
    root = resolve_data_root(data_root)
    pool = collect_split(root, "train") + collect_split(root, "val")
    test = collect_split(root, "test")
    if not pool or not test:
        raise ValueError("Both training and test sets must contain images")
    if check_duplicates:
        pool = deduplicate_pool(pool, duplicate_report)
    train, validation = train_test_split(
        pool, test_size=validation_fraction, random_state=seed,
        stratify=[record.label for record in pool],
    )
    splits = {"train": sorted(train, key=lambda r: str(r.path)),
              "validation": sorted(validation, key=lambda r: str(r.path)),
              "test": test}
    for name, records in splits.items():
        if {record.label for record in records} != {0, 1}:
            raise ValueError(f"{name} must contain both classes")
    paths = [record.path.resolve() for records in splits.values() for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("A file path occurs in more than one split")
    if check_duplicates:
        reject_cross_split_duplicates(splits)
    return splits


def reject_cross_split_duplicates(splits: dict[str, list[ImageRecord]]) -> None:
    """Catch identical image bytes across splits; this cannot detect patient overlap."""
    seen: dict[tuple[int, str], tuple[str, Path]] = {}
    for split_name, records in splits.items():
        for record in records:
            signature = image_signature(record.path)
            if signature in seen and seen[signature][0] != split_name:
                previous_split, previous_path = seen[signature]
                raise ValueError(f"Identical images across {previous_split}/{split_name}: "
                                 f"{previous_path}, {record.path}")
            seen.setdefault(signature, (split_name, record.path))


def split_manifest(splits: dict[str, list[ImageRecord]], root: Path) -> dict:
    return {name: [{"path": str(record.path.relative_to(root)), "label": record.label}
                   for record in records] for name, records in splits.items()}
