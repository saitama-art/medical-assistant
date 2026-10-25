"""根目录 testing/datasets 的结构、兼容接口与版本指纹测试。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acceptance import conditions
from acceptance.cases import fr01_cases
from acceptance.datasets import load_dataset
from acceptance.fr01 import FR01_SAMPLES
from acceptance.health72 import HEALTH72, HEALTH72_VERSION
from acceptance.scenarios import SCENARIOS, SCENARIOS_VERSION


class TestVersionedDatasets(unittest.TestCase):
    def test_legacy_interfaces_and_counts_unchanged(self):
        self.assertEqual(len(HEALTH72), 72)
        self.assertEqual(len(SCENARIOS), 22)
        self.assertEqual(len(FR01_SAMPLES), 8)
        self.assertEqual(HEALTH72_VERSION, "2026-10-21")
        self.assertEqual(SCENARIOS_VERSION, "2026-10-25")
        self.assertTrue(all(isinstance(sample, tuple) and len(sample) == 2
                            for sample in FR01_SAMPLES))

    def test_case_ids_and_order_unchanged(self):
        expected_health = [f"{group}{number:02d}"
                           for group in "ABCDEFGHIJKL" for number in range(1, 7)]
        self.assertEqual([case["id"] for case in HEALTH72], expected_health)
        self.assertEqual([case["id"] for case in SCENARIOS],
                         [f"CS-{number:02d}" for number in range(1, 23)])
        self.assertEqual([case["id"] for case in fr01_cases()],
                         [f"FR-01-{number}" for number in range(1, 9)])
        self.assertEqual([case["question"] for case in fr01_cases()],
                         [question for question, _keyword in FR01_SAMPLES])

    def test_conditions_take_versions_from_dataset_loader(self):
        versions = {"health72": "health-test", "scenarios": "scenarios-test"}
        with patch("acceptance.conditions.promptmeta.prompt_meta", return_value={}), \
                patch("acceptance.conditions.dataset_version", side_effect=versions.get):
            current = conditions.current_conditions([], git_sha="test", stats={})
        self.assertEqual(current["testset_versions"], {
            "coverage": "v2026-10-22",
            "health72": "health-test",
            "scenarios": "scenarios-test",
        })

    def test_loader_accepts_valid_dataset_from_explicit_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {"name": "sample", "version": "v1", "cases": [{"question": "q"}]}
            (root / "sample.json").write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_dataset("sample", root), payload)

    def test_loader_rejects_invalid_common_structure(self):
        invalid_payloads = [
            {"name": "wrong", "version": "v1", "cases": []},
            {"name": "sample", "version": "", "cases": []},
            {"name": "sample", "version": "v1", "cases": ["not-an-object"]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    (root / "sample.json").write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_dataset("sample", root)


if __name__ == "__main__":
    unittest.main()
