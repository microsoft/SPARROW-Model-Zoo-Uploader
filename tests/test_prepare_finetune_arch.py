"""``prepare-finetune --model-arch``: one validated non-ResNet architecture (efficientnet_b0).

Strategy: the existing ResNet-50 contract (``test_prepare_finetune.py``) stays untouched and
keeps pinning the default. These tests add the smallest set of real cases for the second
architecture, driven through the public CLI exactly as an agent runs it:

- ``--model-arch efficientnet_b0`` with real timm efficientnet_b0 tensors writes the same
  ``sparrow-finetune/base-v1`` package with ``model_arch=efficientnet_b0`` and the input
  tensors unchanged;
- the architecture is what the user declares, never inferred from the weights: efficientnet
  weights without the flag are refused against the resnet50 default, and resnet50 weights with
  ``--model-arch efficientnet_b0`` are refused;
- an architecture that has not been validated (convnext_tiny) is refused, and a head whose rows
  differ from the label count is refused; every refusal writes nothing.

Contract consumed independently by the fine-tune service
(``tests/finetune/test_release_model_flows.py`` there).
"""

from __future__ import annotations

import json
import zipfile

import pytest

from test_prepare_finetune import (
    LABELS,
    MEMBERS,
    SCHEMA,
    resnet_state,
    run_prepare,
    save_safetensors,
    sha,
    write_labels,
)

ARCH = "efficientnet_b0"


def run_prepare_all(*argv) -> tuple[int, str]:
    """Like ``run_prepare`` but keeps stderr too: argparse ``choices`` errors land there."""
    import contextlib
    import io

    from sparrow_uploader.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = main(["prepare-finetune", *map(str, argv)])
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 2
    return rc, out.getvalue() + err.getvalue()


@pytest.fixture(scope="module")
def eff() -> dict:
    return resnet_state(ARCH, classes=len(LABELS))  # any timm name; builds random weights offline


@pytest.fixture(scope="module")
def eff_file(eff, tmp_path_factory):
    return save_safetensors(tmp_path_factory.mktemp("eff") / "effnet.safetensors", eff)


@pytest.fixture
def labels_file(tmp_path):
    return write_labels(tmp_path / "labels.txt", LABELS)


def _nothing_written(out_dir) -> None:
    assert list(out_dir.iterdir()) == [], list(out_dir.iterdir())


def test_efficientnet_b0_package_is_the_base_v1_contract(eff, eff_file, labels_file, tmp_path):
    import torch
    from safetensors.torch import load

    out = tmp_path / "pkg.zip"
    before = sha(eff_file.read_bytes())
    rc, data, raw = run_prepare(
        "--weights", eff_file, "--labels", labels_file, "--output", out, "--model-arch", ARCH
    )
    assert rc == 0, raw
    assert data["result"] == "pass" and data["model_arch"] == ARCH, data
    with zipfile.ZipFile(out) as zf:
        infos = zf.infolist()
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        weights = zf.read("weights.safetensors")
    assert sorted(i.filename for i in infos) == MEMBERS
    assert all(i.compress_type == zipfile.ZIP_STORED for i in infos)
    assert manifest == {
        "schema": SCHEMA,
        "pipeline": "conventional_classifier",
        "model_arch": ARCH,
        "labels": LABELS,
        "files": {"weights.safetensors": sha(weights)},
    }
    packaged = load(weights)
    assert set(packaged) == set(eff)
    assert [k for k, v in eff.items() if not torch.equal(packaged[k], v)] == []
    assert sha(eff_file.read_bytes()) == before


@pytest.mark.parametrize(
    "weights_arch, flag, needle",
    [
        # Default stays resnet50: efficientnet weights are not "recognised" into a package.
        (ARCH, None, "resnet50"),
        # Declared efficientnet_b0 but the tensors are resnet50.
        ("resnet50", ARCH, ARCH),
        # Not a validated architecture, even though timm can build it.
        ("convnext_tiny", "convnext_tiny", "convnext_tiny"),
    ],
)
def test_architecture_is_declared_and_validated_never_inferred(
    weights_arch, flag, needle, labels_file, tmp_path
):
    weights = save_safetensors(
        tmp_path / "w.safetensors", resnet_state(weights_arch, classes=len(LABELS))
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    argv = ["--weights", weights, "--labels", labels_file, "--output", out_dir / "pkg.zip"]
    if flag is not None:
        argv += ["--model-arch", flag]
    rc, text = run_prepare_all(*argv)
    assert rc != 0, text
    # A refusal that names the architecture, not argparse rejecting an unknown flag.
    assert needle in text and "unrecognized arguments" not in text, text
    _nothing_written(out_dir)


def test_efficientnet_head_must_match_the_label_count(eff_file, tmp_path):
    labels = write_labels(tmp_path / "labels.txt", [*LABELS, "okapi"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc, text = run_prepare_all(
        "--weights", eff_file, "--labels", labels, "--output", out_dir / "pkg.zip",
        "--model-arch", ARCH,
    )
    assert rc != 0 and "label" in text and "unrecognized arguments" not in text, text
    _nothing_written(out_dir)


def test_agent_skill_documents_the_model_arch_option():
    from pathlib import Path

    skill = Path(__file__).resolve().parents[1] / "skills" / "sparrow-model-uploader"
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(skill.rglob("*.md")))
    assert "--model-arch" in text and ARCH in text
