from __future__ import annotations

import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from validate_canonical_checkpoints import validate_one  # type: ignore  # noqa: E402


EXPECTED_CURRENT_MAIN_PAYLOADS = {
    "cora": ("f5e0ad19a10b007f214a74f4130951ea5a2381a10a510dbc9fe1dace851fc7db", "e5faa44062ba78062ffac5a083da35b03f68e976d6d85de0c01c41cbb4fe3196", "12e9948ae64e808983efe87fa50168461ff4f04d6bca418a68613feee35e7c37"),
    "citeseer": ("af9c413eaec4fc6085afe4f4cfff27d1733acf69341835218696d25fdb6af3f7", "c34468434970cdc707bce238b22d676644f459712ca1454cf70b9eaa0566c43a", "c5f99c8941d786a8c3767511ed13ac292e11a8850d7c66dc46c7474399954605"),
    "pubmed": ("7bb0e976bdfe7b1cf4f8cb3a9076fd5d71e96a2353cac037fcec76c045ed6398", "6b08ba128e4c1c29e6e72dfc65c2c4359c2aaef9b3032f2c1070242535e608ca", "0052bd4cec9c557345262f87e29a3b61bb348a49ab8e5fd2da754b2a176a8deb"),
    "ppi": ("546e070e0c3e0132efe10e2a693dd3de2f8ed9f33227e04d33eb00bf39e52282", "1e7abf4f8468b36f9535a62b1e61bc1bbf327ff5b783c30e22217df0dc6f5062", "42a89ac9c207dd1dae54c2c9ad8350c1463b7caca95cefa92fa427af4541bbbe"),
    "ogbn-arxiv": ("3f4bb23664c1446c49b78404593ef701b15c6e27614829039446dc0daa0dd6c9", "e49c578434376093ea815557faea2b721a9c5369579615b067ce37c8e59bd0f0", "cdc5d2ca46d3b483bd9ed8ab294579b70d78705fc5d96d2e072cdfc6a94a6517"),
}


class CheckpointInstallationTests(unittest.TestCase):
    def test_all_five_installed_payloads_equal_current_main_snapshot(self) -> None:
        for dataset, expected in EXPECTED_CURRENT_MAIN_PAYLOADS.items():
            with self.subTest(dataset=dataset):
                report = validate_one(dataset)
                self.assertEqual(report["tensors_txt_sha256"], expected[0])
                self.assertEqual(report["tensors_npz_sha256"], expected[1])
                self.assertEqual(report["logical_payload_sha256"], expected[2])
                self.assertEqual(report["tensor_count"], 56)

    def test_official_launcher_validates_and_never_converts(self) -> None:
        source = (ROOT / "scripts" / "gat_zkgpt_experiment.py").read_text(encoding="utf-8")
        self.assertIn("validate_canonical_checkpoints.py", source)
        self.assertNotIn("prepare_canonical_checkpoint.py", source)
        self.assertNotIn("convert canonical", source.lower())
        self.assertNotIn("drop_hidden", source)


if __name__ == "__main__":
    unittest.main()
