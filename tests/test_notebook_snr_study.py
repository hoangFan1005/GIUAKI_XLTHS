"""SNR evidence must appear in each standalone notebook build."""
import ast
import math
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from tools import build_notebooks as builder


class NotebookSNRStudyTests(unittest.TestCase):
    def test_each_algorithm_notebook_includes_measured_proxy_and_frozen_noise_sweep(self):
        for algorithm in ("tt1", "tt2", "tt3"):
            with self.subTest(algorithm=algorithm):
                notebook = builder.build_notebook(algorithm)
                markdown = "\n".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"] if cell["cell_type"] == "markdown"
                )
                code = "\n".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"] if cell["cell_type"] == "code"
                )
                self.assertIn("SNR proxy", markdown)
                self.assertIn("phone vs studio", markdown)
                self.assertIn("only two WAVs", markdown)
                self.assertIn("synthetic-noise", markdown.lower())
                self.assertIn("10 * math.log10(powers[1] / powers[0])", code)
                self.assertIn("SNR_PROXY_COMPARISON", code)
                self.assertIn('axes[0].set_ylabel("SNR proxy (dB)")', code)
                self.assertIn("axis.axhline", code)
                self.assertIn("SYNTHETIC_NOISE_STUDY_ROWS", code)
                self.assertIn("SYNTHETIC_NOISE_LEVELS_DB = (20, 10, 0)", code)
                self.assertIn("assert_model_locked()", code)
                self.assertIn("FROZEN_MODEL_DIGESTS_BEFORE_SNR_STUDY", code)
                self.assertNotIn("fit_training_model(", code[code.index("SYNTHETIC_NOISE_STUDY_ROWS"):])
                ast.parse(code)

                proxy_cell = next(
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                    and "SNR_PROXY_COMPARISON = []" in "".join(cell["source"])
                )
                def record(name, split):
                    return dict(name=name, split=split, sample_rate=1, duration=4,
                                samples=[1.0, 1.0, 2.0, 2.0],
                                intervals=[(0.0, 2.0, "sil"), (2.0, 4.0, "v")])

                shown_tables = []
                namespace = dict(
                    math=math,
                    TRAIN_RECORDS=[record("phone_train", "train"), record("studio_train", "train")],
                    TEST_RECORDS=[record("phone_test", "test"), record("studio_test", "test")],
                    ground_truth_speech_envelope=lambda intervals: (2.0, 4.0),
                    display_table=lambda rows, columns, caption: shown_tables.append((rows, caption)),
                    plt=plt,
                )
                with patch("matplotlib.pyplot.show"):
                    exec(compile(proxy_cell, "<snr-proxy-cell>", "exec"), namespace)
                plt.close("all")
                self.assertEqual(len(namespace["SNR_PROXY_ROWS"]), 4)
                self.assertAlmostEqual(namespace["SNR_PROXY_ROWS"][0]["snr_proxy_db"], 10 * math.log10(4))
                self.assertEqual(len(namespace["SNR_PROXY_COMPARISON"]), 4)
                self.assertTrue(all(row["files"] == 1 for row in namespace["SNR_PROXY_COMPARISON"]))


if __name__ == "__main__":
    unittest.main()
