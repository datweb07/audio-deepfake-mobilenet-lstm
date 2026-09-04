"""Verify completed full clean plus diagnostic robustness; never invokes training."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.lava5 import DEFAULT_OUTPUT, MODELS, read_csv, sha256, verify_protocol
from benchmark.lava5_report import load_result
from src.lava.artifacts import write_json_atomic


def check(output, samples=100):
    output = Path(output)
    subset = output / f"diagnostic_{samples}"
    protocol = verify_protocol(output)
    verify_protocol(subset)
    canonical = read_csv(output / "protocol/test_samples.csv")
    selected = read_csv(subset / "protocol/test_samples.csv")
    assert len(selected) == samples
    assert len({r["sample_id"] for r in selected}) == samples
    expected = {r["sample_id"]: r for r in canonical}
    assert all(r["sample_id"] in expected and r["label"] == expected[r["sample_id"]]["label"] for r in selected)
    suites = {"noise": ["snr_20", "snr_10", "snr_5", "snr_0"],
              "compression": ["mp3_128k", "mp3_64k", "opus_64k", "aac_96k"],
              "replay": ["synthetic_channel"]}
    for suite, conditions in suites.items():
        for condition in conditions:
            manifest = read_csv(subset / "protocol/conditions" / suite / f"{condition}.csv")
            assert [r["sample_id"] for r in manifest] == [r["sample_id"] for r in selected]
            assert [r["label"] for r in manifest] == [r["label"] for r in selected]
            assert all(sha256(subset / r["path"].replace("\\", "/")) == r["sha256"] for r in manifest)
    stress_count = 0
    for name in MODELS:
        clean = load_result(output / "clean" / name, canonical)
        paired = load_result(subset / "clean" / name, selected, diagnostic=True)
        assert clean is not None and paired is not None
        original = {r["sample_id"]: r for r in clean["rows"]}
        assert all(r == original[r["sample_id"]] for r in paired["rows"])
        load = json.loads((output / "protocol" / f"{name}_load.json").read_text())
        assert load["load_status"] == load["adapter_parity"] == "PASS"
        for suite, conditions in suites.items():
            for condition in conditions:
                result = load_result(subset / "robustness" / suite / condition / name, selected, diagnostic=True)
                assert result is not None
                assert result["summary"]["threshold"] == clean["summary"]["threshold"]
                stress_count += 1
        measured_path = output / "efficiency" / name / "summary.json"
        measured = json.loads(measured_path.read_text())
        assert measured["status"] == "BENCHMARKED"
        assert measured["warmup"] >= 10 and measured["runs"] >= 50
        assert measured["batch_size"] == 1 and measured["device"] == "CPU"
        for component in ("model_only", "preprocessing", "end_to_end"):
            assert len(measured[component]["runs_ms"]) == measured["runs"]
        inherited = json.loads((subset / "efficiency" / name / "summary.json").read_text())
        assert inherited["inherited_sha256"] == sha256(measured_path)
    for name in ("rawnet2", "aasist"):
        audit = json.loads((output / "protocol" / f"{name}_native_audit.json").read_text())
        assert audit["status"] == "PASS" and audit["parity_samples"] >= 2
    for directory in (output, subset):
        rows = read_csv(directory / "lava_5_results.csv")
        assert {r["Model"] for r in rows} == set(MODELS)
        stamp = json.loads((directory / "error_analysis/statistics_state.json").read_text())
        assert stamp["iterations"] == 1000
    official = json.loads((output / "report/acceptance.json").read_text())
    diagnostic = json.loads((subset / "report/acceptance.json").read_text())
    assert not official["full_acceptance"] and not diagnostic["full_acceptance"]
    assert diagnostic["status"] == "LAVA-5 DIAGNOSTIC SUBSET COMPLETE"
    assert not (output / "pareto/pareto_results.csv").exists()
    assert len(read_csv(subset / "pareto/pareto_results.csv")) == 5
    result = dict(status="PASS", scope="FULL_CLEAN_AND_DIAGNOSTIC_ROBUSTNESS",
                  clean_samples_per_model=len(canonical), diagnostic_samples_per_condition=samples,
                  stress_model_condition_results=stress_count, models=list(MODELS),
                  artifact_manifest_seals="PASS", thresholds_unchanged=True,
                  full_test_robustness="NOT_RUN", official_pareto="NOT_RUN",
                  exploratory_diagnostic_pareto="PASS", no_retraining=True,
                  shuffle_excluded=True, full_lava5_complete=False)
    write_json_atomic(output / "report/diagnostic_scope_acceptance.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    check(args.output, args.samples)
