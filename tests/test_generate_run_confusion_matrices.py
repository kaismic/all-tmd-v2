import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "generate-run-confusion-matrices.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_run_confusion_matrices", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_artifacts(
    results_root: Path,
    run_id: str,
    matrix: list[list[int]] | None = None,
) -> Path:
    artifacts_dir = (
        results_root
        / "download-a"
        / "mlflow"
        / "mlartifacts"
        / "1"
        / run_id
        / "artifacts"
    )
    artifacts_dir.mkdir(parents=True)
    metrics = {
        "collector_holdout": {
            "classification_report": {
                "bus": {"support": 10},
                "car": {"support": 10},
                "accuracy": 0.75,
                "macro avg": {"support": 20},
                "weighted avg": {"support": 20},
            },
            "confusion_matrix": matrix or [[8, 2], [3, 7]],
        }
    }
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    return artifacts_dir


def test_find_run_artifacts_matches_exact_mlflow_run_directories(tmp_path):
    first_id = "a" * 32
    second_id = "b" * 32
    first = _write_artifacts(tmp_path, first_id)
    _write_artifacts(tmp_path, second_id)
    duplicate_metrics = tmp_path / "download-a" / "work" / first_id / "metrics.json"
    duplicate_metrics.parent.mkdir(parents=True)
    duplicate_metrics.write_text("{}", encoding="utf-8")

    matches = MODULE.find_run_artifacts(tmp_path, [first_id, "missing"])

    assert matches == {first_id: [first], "missing": []}


def test_build_figure_normalizes_rows_and_uses_condition_title():
    figure = MODULE.build_figure(
        [[8, 2], [0, 0]],
        ["bus", "car"],
        "e821edccef3648d1be52848dd413f007",
        "A",
    )
    try:
        axis = figure.axes[0]
        np.testing.assert_allclose(
            np.asarray(axis.images[0].get_array()),
            [[0.8, 0.2], [0.0, 0.0]],
        )
        assert axis.get_title() == "Condition A (row normalized)"
    finally:
        figure.clear()


def test_main_generates_standard_artifact_and_reports_missing_run(
    tmp_path, capsys
):
    run_id = "0c8015ecc7d541cf95db0b40c7a581e7"
    _write_artifacts(tmp_path, run_id)

    exit_code = MODULE.main([run_id, "missing"], results_root=tmp_path)

    output_path = tmp_path / "confusion-matrices" / MODULE.output_filename(run_id)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert output_path.is_file()
    assert output_path.name == (
        "conf-matrix-norm-0c8015e.png"
    )
    assert str(output_path) in captured.out
    assert "run ID not found: missing" in captured.err


def test_main_accepts_a_custom_output_directory(tmp_path):
    run_id = "e821edccef3648d1be52848dd413f007"
    _write_artifacts(tmp_path, run_id)
    output_dir = tmp_path / "collected-images"

    exit_code = MODULE.main(
        [run_id, "--output-dir", str(output_dir)],
        results_root=tmp_path,
    )

    assert exit_code == 0
    assert (output_dir / MODULE.output_filename(run_id)).is_file()


def test_main_uses_condition_in_title_and_filename(tmp_path):
    run_id = "e821edccef3648d1be52848dd413f007"
    _write_artifacts(tmp_path, run_id)
    output_dir = tmp_path / "collected-images"
    condition_map = tmp_path / "conditions.json"
    condition_map.write_text(
        json.dumps(
            {"conditions": [{"condition": "A", "run_id": "e821edc"}]}
        ),
        encoding="utf-8",
    )

    exit_code = MODULE.main(
        [
            run_id,
            "--output-dir",
            str(output_dir),
            "--condition-map",
            str(condition_map),
        ],
        results_root=tmp_path,
    )

    assert exit_code == 0
    assert (output_dir / "conf-matrix-norm-e821edc-A.png").is_file()
