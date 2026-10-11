"""Weights trained with Ultralytics default to AGPL-3.0 (Ultralytics' licence covers derived models)."""

import json

from sparrow_uploader.compliance import provenance_issues

from conftest import INIT


def _no_license(extra=()):
    argv = list(INIT)
    i = argv.index("--license")
    del argv[i : i + 2]
    return argv + list(extra)


def _init(run, *argv):
    return run("init", "--model-id", "yolo-det", "--task", "detector", "--domain", "camera_trap", *argv)


def test_ultralytics_without_license_defaults_to_agpl(run):
    rc, out = _init(run, *_no_license(["--framework", "pytorch 2.6 / ultralytics 8.3"]))
    assert rc == 0, out
    prov = json.loads((run.ws / "yolo-det" / "PROVENANCE.json").read_text())
    assert prov["license"] == "AGPL-3.0"
    assert any("AGPL-3.0" in w and "default" in w for w in out["warnings"]), out


def test_missing_license_without_ultralytics_is_refused(run):
    rc, out = _init(run, *_no_license(["--framework", "pytorch 2.6 / timm 1.0"]))
    assert rc != 0, out


def test_ultralytics_with_other_license_warns(run):
    argv = _no_license(["--framework", "ultralytics 8.3", "--license", "CC-BY-4.0"])
    rc, out = _init(run, *argv)
    assert rc == 0, out
    assert any("Enterprise" in w for w in out["warnings"]), out


def test_lint_provenance_warns_on_non_agpl_ultralytics():
    base = {"source": "https://x.org/w", "license_source_url": "https://x.org/l",
            "source_weights": [1], "source_revision": "r", "display_name": "d"}
    _, warns = provenance_issues({**base, "framework": "ultralytics 8.3", "license": "CC-BY-4.0"})
    assert any("Enterprise" in w for w in warns)
    _, warns = provenance_issues({**base, "framework": "ultralytics 8.3", "license": "AGPL-3.0"})
    assert not any("Enterprise" in w for w in warns)
