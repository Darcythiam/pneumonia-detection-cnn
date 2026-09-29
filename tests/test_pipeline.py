import tempfile
from pathlib import Path
import unittest

from pneumonia_cnn.evaluation import choose_threshold, evaluate
from pneumonia_cnn.splits import CLASS_NAMES, make_splits, resolve_data_root
from pneumonia_cnn.train import summarize


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "chest_xray"
        for split, count in (("train", 8), ("val", 2), ("test", 2)):
            for name in CLASS_NAMES:
                folder = self.root / split / name
                folder.mkdir(parents=True)
                for i in range(count):
                    (folder / f"{i}.jpeg").write_bytes(f"{split}:{name}:{i}".encode())

    def test_stratified_splits_are_deterministic_and_keep_test_separate(self):
        first = make_splits(self.root, validation_fraction=0.2, seed=7)
        second = make_splits(self.root, validation_fraction=0.2, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first["test"]), 4)
        self.assertEqual({r.label for r in first["validation"]}, {0, 1})
        self.assertTrue(all("test" in r.path.parts for r in first["test"]))
        self.assertTrue(set(r.path for r in first["train"]).isdisjoint(
            r.path for r in first["validation"]))

    def test_identical_bytes_across_train_and_test_are_rejected(self):
        (self.root / "test" / "NORMAL" / "0.jpeg").write_bytes(
            (self.root / "train" / "NORMAL" / "0.jpeg").read_bytes())
        with self.assertRaisesRegex(ValueError, "Identical images"):
            make_splits(self.root)

    def test_identical_training_images_are_removed_before_splitting(self):
        duplicate = self.root / "train" / "PNEUMONIA" / "1.jpeg"
        original = self.root / "train" / "PNEUMONIA" / "0.jpeg"
        duplicate.write_bytes(original.read_bytes())
        removed = []
        splits = make_splits(self.root, seed=7, duplicate_report=removed)
        self.assertEqual(removed, [(duplicate, original)])
        self.assertEqual(sum(len(splits[k]) for k in ("train", "validation")), 19)
        self.assertNotIn(duplicate, [r.path for k in ("train", "validation")
                                     for r in splits[k]])
        self.assertEqual(make_splits(self.root, seed=7), splits)

    def test_identical_files_with_conflicting_labels_are_rejected(self):
        (self.root / "train" / "PNEUMONIA" / "1.jpeg").write_bytes(
            (self.root / "train" / "NORMAL" / "1.jpeg").read_bytes())
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            make_splits(self.root)

    def test_nested_layout_is_supported(self):
        self.assertEqual(resolve_data_root(self.root.parent), self.root)


class EvaluationTests(unittest.TestCase):
    def test_threshold_is_selected_before_test_evaluation(self):
        validation_labels = [0, 0, 1, 1]
        validation_probabilities = [0.1, 0.4, 0.6, 0.9]
        threshold = choose_threshold(validation_labels, validation_probabilities)
        self.assertAlmostEqual(threshold, 0.6)
        test_report = evaluate([0, 1], [0.55, 0.65], threshold)
        self.assertEqual(test_report["accuracy"], 1.0)
        self.assertEqual(test_report["confusion_matrix"], [[1, 0], [0, 1]])

    def test_metrics_reject_invalid_probabilities_or_one_class(self):
        with self.assertRaises(ValueError):
            choose_threshold([0, 0], [0.2, 0.3])
        with self.assertRaises(ValueError):
            evaluate([0, 1], [0.3, 1.2], 0.5)

    def test_perfect_calibration_has_zero_brier_and_ece(self):
        report = evaluate([0, 1], [0, 1], 0.5)
        self.assertEqual(report["brier_score"], 0.0)
        self.assertEqual(report["ece_10_bins"], 0.0)
        self.assertEqual(sum(row["count"] for row in report["reliability_bins"]), 2)

    def test_aggregate_retains_each_seeds_results(self):
        runs = [{"candidates": {"baseline": {"test": {
            metric: value for metric in ("accuracy", "roc_auc", "average_precision",
                                          "brier_score", "ece_10_bins", "pneumonia_recall",
                                          "normal_specificity")}}}} for value in (0.5, 1.0)]
        aggregated = summarize(runs, ["baseline"])
        self.assertEqual(aggregated["baseline"]["accuracy"]["values"], [0.5, 1.0])
        self.assertEqual(aggregated["baseline"]["accuracy"]["mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
