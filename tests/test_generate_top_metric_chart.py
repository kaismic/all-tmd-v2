import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "generate-top-metric-chart.py"
)
SPEC = importlib.util.spec_from_file_location("generate_top_metric_chart", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_metrics(
    results_root: Path,
    *,
    download_id: str,
    run_id: str,
    accuracy: float,
    macro_f1: float,
    balanced_accuracy: float,
) -> None:
    metrics_dir = (
        results_root
        / download_id
        / "mlflow"
        / "mlartifacts"
        / "1"
        / run_id
        / "artifacts"
    )
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(
            {
                "collector_holdout": {
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "balanced_accuracy": balanced_accuracy,
                }
            }
        ),
        encoding="utf-8",
    )


def _write_condition_map(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "conditions": [
                    {"condition": "A", "run_id": "abcdefg"},
                    {"condition": "B", "run_id": "7654321"},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_collects_unique_runs_and_ranks_selected_metric(tmp_path):
    _write_metrics(
        tmp_path,
        download_id="download-one",
        run_id="bbbbbbb2222222222222222222222222",
        accuracy=0.80,
        macro_f1=0.91,
        balanced_accuracy=0.83,
    )
    _write_metrics(
        tmp_path,
        download_id="download-one",
        run_id="aaaaaaa1111111111111111111111111",
        accuracy=0.95,
        macro_f1=0.85,
        balanced_accuracy=0.90,
    )
    _write_metrics(
        tmp_path,
        download_id="download-two",
        run_id="bbbbbbb2222222222222222222222222",
        accuracy=0.80,
        macro_f1=0.91,
        balanced_accuracy=0.83,
    )

    results = MODULE.collect_run_metrics(
        tmp_path, "collector_holdout.macro_f1"
    )

    assert [(result.run_id[:7], result.value) for result in results] == [
        ("bbbbbbb", 0.91),
        ("aaaaaaa", 0.85),
    ]


def test_build_figure_uses_short_run_ids_and_mapped_title():
    results = [
        MODULE.RunMetric("abcdefg123", 0.93, Path("first.json")),
        MODULE.RunMetric("7654321abc", 0.89, Path("second.json")),
    ]

    figure = MODULE.build_figure(results, "collector_holdout.accuracy")
    try:
        axis = figure.axes[0]
        assert axis.get_title() == "Best Overall Accuracy"
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "abcdefg",
            "7654321",
        ]
        assert [bar.get_height() for bar in axis.patches] == [0.93, 0.89]
        assert len({bar.get_facecolor() for bar in axis.patches}) == 2
    finally:
        figure.clear()


def test_build_combined_figure_uses_condition_labels():
    first = [MODULE.RunMetric("abcdefg123", 0.93, Path("first.json"))]
    second = [MODULE.RunMetric("7654321abc", 0.89, Path("second.json"))]

    figure = MODULE.build_combined_figure(
        [
            ("collector_holdout.balanced_accuracy", first),
            ("collector_holdout.macro_f1", second),
        ],
        {"abcdefg": "A", "7654321": "B"},
    )
    try:
        assert [axis.get_title() for axis in figure.axes] == [
            "Best Balanced Accuracy",
            "Best Macro F1 Score",
        ]
        assert [axis.get_xticklabels()[0].get_text() for axis in figure.axes] == [
            "A",
            "B",
        ]
        assert all(axis.get_xlabel() == "Condition" for axis in figure.axes)
    finally:
        figure.clear()


def test_main_generates_metric_and_limit_filename(tmp_path, capsys):
    results_root = tmp_path / "aws-results"
    output_dir = tmp_path / "charts"
    _write_metrics(
        results_root,
        download_id="download-one",
        run_id="abcdefg1234567890123456789012345",
        accuracy=0.94,
        macro_f1=0.92,
        balanced_accuracy=0.91,
    )

    exit_code = MODULE.main(
        ["collector_holdout.balanced_accuracy", "3"],
        results_root=results_root,
        output_dir=output_dir,
    )

    output_path = output_dir / "collector_holdout.balanced_accuracy-top-3.png"
    assert exit_code == 0
    assert output_path.is_file()
    assert str(output_path) in capsys.readouterr().out


def test_main_combines_metrics_and_loads_condition_map(tmp_path, capsys):
    results_root = tmp_path / "aws-results"
    output_dir = tmp_path / "charts"
    condition_map = tmp_path / "conditions.json"
    _write_condition_map(condition_map)
    _write_metrics(
        results_root,
        download_id="download-one",
        run_id="abcdefg1234567890123456789012345",
        accuracy=0.94,
        macro_f1=0.92,
        balanced_accuracy=0.91,
    )

    exit_code = MODULE.main(
        [
            "collector_holdout.balanced_accuracy",
            "3",
            "--combine-with",
            "collector_holdout.macro_f1",
            "--condition-map",
            str(condition_map),
        ],
        results_root=results_root,
        output_dir=output_dir,
    )

    output_path = (
        output_dir
        / "collector_holdout.balanced_accuracy-and-macro_f1-top-3.png"
    )
    assert exit_code == 0
    assert output_path.is_file()
    assert str(output_path) in capsys.readouterr().out


def test_main_uses_dedicated_results_subdirectory_by_default(tmp_path):
    results_root = tmp_path / "aws-results"
    _write_metrics(
        results_root,
        download_id="download-one",
        run_id="abcdefg1234567890123456789012345",
        accuracy=0.94,
        macro_f1=0.92,
        balanced_accuracy=0.91,
    )

    exit_code = MODULE.main(
        ["collector_holdout.accuracy", "1"],
        results_root=results_root,
    )

    output_path = (
        results_root
        / "top-metric-charts"
        / "collector_holdout.accuracy-top-1.png"
    )
    assert exit_code == 0
    assert output_path.is_file()


def test_main_reports_when_no_downloaded_metrics_exist(tmp_path, capsys):
    exit_code = MODULE.main(
        ["collector_holdout.accuracy", "2"],
        results_root=tmp_path,
        output_dir=tmp_path,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "no downloaded MLflow metrics.json files" in captured.err
