"""Compare models over fixed seeds; choose thresholds and deployment on validation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from .evaluation import choose_threshold, evaluate
from .splits import make_splits, resolve_data_root, split_manifest


DEGRADATIONS = ("noise", "blur", "low_contrast")
METRICS = ("accuracy", "roc_auc", "average_precision", "brier_score",
           "ece_10_bins", "pneumonia_recall", "normal_specificity")


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(runs, models):
    return {
        name: {
            metric: {"mean": float(np.mean(values)),
                     "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                     "values": values}
            for metric in METRICS
            for values in [[run["candidates"][name]["test"][metric] for run in runs]]
        }
        for name in models
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--download-kaggle", action="store_true",
                        help="Download/cache paultimothymooney/chest-xray-pneumonia with KaggleHub")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="+", choices=("baseline", "resnet50"),
                        default=["baseline", "resnet50"])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--skip-duplicate-check", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs/batch-size must be positive, image-size must be >= 32")
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Model names and seeds must be unique")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be nonnegative")
    if args.download_kaggle and args.data_root != Path("data/chest_xray"):
        parser.error("Use either --download-kaggle or --data-root")
    if args.download_kaggle:
        import kagglehub
        dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"KaggleHub dataset path: {dataset_path}", flush=True)
    else:
        dataset_path = args.data_root
    root = resolve_data_root(dataset_path)
    output = (args.output_dir or Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    splits_by_seed = {}
    duplicate_reports = {}
    for seed in args.seeds:
        removed = []
        splits_by_seed[seed] = make_splits(
            root, validation_fraction=args.validation_fraction, seed=seed,
            check_duplicates=not args.skip_duplicate_check, duplicate_report=removed)
        duplicate_reports[seed] = removed
    output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import tensorflow as tf
    from .data import make_dataset
    from .models import build_model  # Also registers custom preprocessing on load.

    runs = []
    for seed in args.seeds:
        splits = splits_by_seed[seed]
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        write_json(seed_dir / "split_manifest.json", split_manifest(splits, root))
        write_json(seed_dir / "duplicate_report.json", {
            "removed_count": len(duplicate_reports[seed]),
            "removed": [{"discarded": str(discarded.relative_to(root)),
                         "retained": str(retained.relative_to(root))}
                        for discarded, retained in duplicate_reports[seed]],
        })
        train_ds = make_dataset(splits["train"], image_size=args.image_size,
                                batch_size=args.batch_size, training=True, seed=seed)
        val_ds = make_dataset(splits["validation"], image_size=args.image_size,
                              batch_size=args.batch_size)
        counts = Counter(record.label for record in splits["train"])
        total = len(splits["train"])
        weights = {label: total / (2 * counts[label]) for label in (0, 1)}
        val_labels = [record.label for record in splits["validation"]]
        candidates = {}
        for name in args.models:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = build_model(name, image_size=args.image_size, seed=seed)
            checkpoint = seed_dir / f"{name}.keras"
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_auc",
                                                   mode="max", save_best_only=True),
                tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                 patience=3, restore_best_weights=True),
            ]
            history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                                class_weight=weights, callbacks=callbacks, verbose=2, shuffle=False)
            write_json(seed_dir / f"{name}_history.json",
                       {key: [float(v) for v in values] for key, values in history.history.items()})
            best = tf.keras.models.load_model(checkpoint)
            probabilities = best.predict(val_ds, verbose=0).reshape(-1)
            threshold = choose_threshold(val_labels, probabilities)
            candidates[name] = {"model_file": str(checkpoint.relative_to(output)),
                                "validation": evaluate(val_labels, probabilities, threshold)}
            del model, best

        # Every model and seed uses its frozen validation threshold on the same test set.
        test_labels = [record.label for record in splits["test"]]
        for name, candidate in candidates.items():
            model = tf.keras.models.load_model(output / candidate["model_file"])
            threshold = candidate["validation"]["threshold"]
            test_ds = make_dataset(splits["test"], image_size=args.image_size,
                                   batch_size=args.batch_size)
            candidate["test"] = evaluate(test_labels, model.predict(test_ds, verbose=0).reshape(-1),
                                         threshold)
            candidate["degradations"] = {}
            for degradation in DEGRADATIONS:
                ds = make_dataset(splits["test"], image_size=args.image_size,
                                  batch_size=args.batch_size, seed=seed, degradation=degradation)
                candidate["degradations"][degradation] = evaluate(
                    test_labels, model.predict(ds, verbose=0).reshape(-1), threshold)
            del model
        run = {"seed": seed, "candidates": candidates,
               "exact_training_duplicates_removed": len(duplicate_reports[seed]),
               "class_counts": {key: dict(Counter(r.label for r in records))
                                for key, records in splits.items()},
               "class_weights": weights}
        runs.append(run)
        write_json(seed_dir / "report.json", run)

    selected_run, selected_model = max(
        ((run, name) for run in runs for name in args.models),
        key=lambda pair: (pair[0]["candidates"][pair[1]]["validation"]["roc_auc"],
                          -pair[0]["seed"], pair[1]))
    selected_seed = selected_run["seed"]
    report = {
        "selection": "Highest validation ROC AUC across seeds/models; lowest seed then name breaks ties",
        "selected_seed": selected_seed, "selected_model": selected_model,
        "candidates": selected_run["candidates"], "runs": runs,
        "aggregate_test": summarize(runs, args.models),
        "test": selected_run["candidates"][selected_model]["test"],
        "data_root": str(root),
        "config": {"seeds": args.seeds, "validation_fraction": args.validation_fraction,
                   "epochs": args.epochs, "batch_size": args.batch_size,
                   "image_size": args.image_size,
                   "degradations": {"noise": "Gaussian sigma=12 on 0-255 pixels",
                                    "blur": "3x3 mean filter",
                                    "low_contrast": "0.65 contrast about 127.5"},
                   "duplicate_check": not args.skip_duplicate_check},
        "versions": {"tensorflow": tf.__version__},
    }
    if not args.skip_onnx:
        from .export import export_onnx
        model_file = output / report["candidates"][selected_model]["model_file"]
        val_ds = make_dataset(splits_by_seed[selected_seed]["validation"],
                              image_size=args.image_size, batch_size=args.batch_size)
        sample_pixels = next(iter(val_ds))[0][:2].numpy()
        report["onnx"] = export_onnx(model_file, output / "selected.onnx", sample_pixels)
    write_json(output / "report.json", report)
    print(f"Selected {selected_model} at seed {selected_seed}; report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
