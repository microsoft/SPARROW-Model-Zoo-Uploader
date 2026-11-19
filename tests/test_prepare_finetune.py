"""``sparrow-uploader prepare-finetune``: a private, offline ResNet-50 training package.

What the command must prove before it reports success, and what these tests pin:

- Contract with the Sparrow fine-tune service (``POST /v1/base-models/local`` there, pinned
  independently by its ``tests/finetune/test_local_base_models.py``): the output ZIP holds
  exactly ``manifest.json`` and ``weights.safetensors``, both ZIP_STORED; the manifest is
  ``schema=sparrow-finetune/base-v1``, ``pipeline=conventional_classifier``,
  ``model_arch=resnet50``, the labels in file order, and the sha256 of the weights member.
- The weights in the package are the input tensors, unchanged, and the input file is not
  modified (the CPU training step that proves the weights train runs on a copy).
- Strict architecture match against timm ``resnet50``: missing, extra or reshaped tensors,
  another architecture, or a head whose size differs from the label count are refused.
- A weights file that is not safetensors (a pickle, a torch checkpoint) is refused without
  being unpickled: a ``__reduce__`` payload must never run.
- Labels are UTF-8, one per line, order kept; empty, blank, duplicate or non-UTF-8 lists are
  refused.
- The command runs with the network blocked, writes nothing but the output ZIP, and writes
  nothing at all when it refuses.

The tests drive ``main([...])`` as an agent does and read its JSON stdout. They build a real
randomly initialised timm resnet50 (no download), so they need torch, timm and safetensors;
when those are missing the tests fail with that message instead of skipping silently.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pickle
import socket
import zipfile
from pathlib import Path

import pytest

from sparrow_uploader.cli import main

SCHEMA = "sparrow-finetune/base-v1"
# Not sorted and not ASCII: the package must keep the file order and the UTF-8 text.
LABELS = ["zebra", "café", "ñandú"]
MEMBERS = ["manifest.json", "weights.safetensors"]


# --------------------------------------------------------------------------- builders


def _stack():
    try:
        import safetensors.torch  # noqa: F401
        import timm
        import torch
    except ImportError as exc:
        pytest.fail(
            f"prepare-finetune tests need torch, timm and safetensors ({exc}); "
            "install the uploader's fine-tune dependencies",
            pytrace=False,
        )
    return torch, timm


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resnet_state(arch: str = "resnet50", classes: int = 3, seed: int = 0) -> dict:
    torch, timm = _stack()
    torch.manual_seed(seed)
    model = timm.create_model(arch, pretrained=False, num_classes=classes)
    return {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}


def save_safetensors(path: Path, state: dict) -> Path:
    from safetensors.torch import save_file

    save_file(state, str(path))
    return path


def write_labels(path: Path, labels) -> Path:
    path.write_bytes(("\n".join(labels) + "\n").encode("utf-8"))
    return path


def run_prepare(*argv: str) -> tuple[int, dict | None, str]:
    """Run the CLI; ``(rc, parsed stdout or None, raw stdout)``. argparse exits are an rc too."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        try:
            rc = main(["prepare-finetune", *map(str, argv)])
        except SystemExit as exc:  # unknown command or bad flags
            rc = exc.code if isinstance(exc.code, int) else 2
    raw = buf.getvalue()
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    return rc, data, raw


class _Plant:
    """Unpickling this creates ``marker``: proof that a loader executed pickle code."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (open, (str(self.marker), "w"))


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def r50() -> dict:
    return resnet_state("resnet50", classes=len(LABELS))


@pytest.fixture(scope="module")
def r50_file(r50, tmp_path_factory) -> Path:
    return save_safetensors(tmp_path_factory.mktemp("in") / "resnet50.safetensors", r50)


@pytest.fixture(scope="module")
def prepared(r50_file, tmp_path_factory) -> dict:
    """One successful run shared by the contract tests (a resnet50 run costs seconds)."""
    labels = write_labels(tmp_path_factory.mktemp("labels") / "labels.txt", LABELS)
    out_dir = tmp_path_factory.mktemp("out")
    before = sha(r50_file.read_bytes())
    rc, data, raw = run_prepare(
        "--weights", r50_file, "--labels", labels, "--output", out_dir / "pkg.zip"
    )
    return {
        "rc": rc,
        "data": data,
        "raw": raw,
        "zip": out_dir / "pkg.zip",
        "input_sha_before": before,
        "input_sha_after": sha(r50_file.read_bytes()),
    }


def _ok(prepared) -> zipfile.ZipFile:
    assert prepared["rc"] == 0, prepared["raw"]
    assert prepared["data"] is not None, prepared["raw"]
    assert prepared["data"].get("result") not in ("fail", "error"), prepared["data"]
    assert prepared["zip"].is_file()
    return zipfile.ZipFile(prepared["zip"])


# --------------------------------------------------------------------------- package contract


def test_package_has_exactly_manifest_and_weights_stored(prepared):
    with _ok(prepared) as zf:
        infos = zf.infolist()
    assert sorted(i.filename for i in infos) == MEMBERS
    assert len(infos) == len(MEMBERS)  # no duplicate member names
    assert all(i.compress_type == zipfile.ZIP_STORED for i in infos), [
        (i.filename, i.compress_type) for i in infos
    ]


def test_manifest_is_the_base_v1_contract(prepared):
    with _ok(prepared) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        weights = zf.read("weights.safetensors")
    assert manifest == {
        "schema": SCHEMA,
        "pipeline": "conventional_classifier",
        "model_arch": "resnet50",
        "labels": LABELS,
        "files": {"weights.safetensors": sha(weights)},
    }


def test_packaged_weights_are_the_input_tensors_and_input_is_untouched(prepared, r50):
    import torch
    from safetensors.torch import load

    with _ok(prepared) as zf:
        packaged = load(zf.read("weights.safetensors"))
    assert set(packaged) == set(r50)
    changed = [k for k, v in r50.items() if not torch.equal(packaged[k], v)]
    assert changed == [], f"the training step leaked into the package: {changed[:5]}"
    assert prepared["input_sha_after"] == prepared["input_sha_before"]


def test_prepare_is_offline_and_writes_only_the_package(r50_file, tmp_path, monkeypatch):
    def no_network(*_a, **_k):
        raise OSError("network blocked by the test: prepare-finetune must be offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    labels = write_labels(tmp_path / "labels.txt", LABELS)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc, data, raw = run_prepare(
        "--weights", r50_file, "--labels", labels, "--output", out_dir / "pkg.zip"
    )
    assert rc == 0, raw
    # No ONNX export, no scratch files beside the package.
    assert sorted(p.name for p in out_dir.iterdir()) == ["pkg.zip"]


# --------------------------------------------------------------------------- refusals


def _weights_case(case: str, tmp_path: Path, request) -> tuple[Path, list[str]]:
    """``(weights file, labels)`` for one malformed or unsupported input."""
    labels = list(LABELS)
    path = tmp_path / "weights.safetensors"
    if case == "not_safetensors":
        path.write_bytes(b"this is not a safetensors file at all")
    elif case == "truncated_header":
        path.write_bytes(b"\xff\xff\xff\xff\xff\xff\x00\x00{")  # header length >> file size
    elif case == "missing_file":
        path = tmp_path / "does-not-exist.safetensors"
    else:
        r50 = request.getfixturevalue("r50")
        if case == "resnet18_weights":
            state = resnet_state("resnet18", classes=len(LABELS))
        elif case == "missing_key":
            state = {k: v for k, v in r50.items() if k != "layer4.2.conv3.weight"}
        elif case == "extra_key":
            import torch

            state = {**r50, "extra.weight": torch.zeros(1)}
        elif case == "wrong_shape":
            import torch

            state = {**r50, "conv1.weight": torch.zeros(64, 3, 3, 3)}
        elif case == "head_label_mismatch":
            state, labels = r50, [*LABELS, "okapi"]
        elif case == "nonfinite":
            import torch

            state = {**r50, "fc.weight": torch.full_like(r50["fc.weight"], float("nan"))}
        else:  # pragma: no cover - parametrize typo
            raise AssertionError(case)
        save_safetensors(path, state)
    return path, labels


@pytest.mark.parametrize(
    "case, needle",
    [
        ("not_safetensors", "safetensors"),
        ("truncated_header", "safetensors"),
        ("missing_file", ""),
        ("resnet18_weights", "resnet50"),
        ("missing_key", "layer4.2.conv3.weight"),
        ("extra_key", "extra.weight"),
        ("wrong_shape", "conv1.weight"),
        ("head_label_mismatch", "label"),
        # The real CPU training step must refuse weights that cannot train (NaN loss).
        ("nonfinite", ""),
    ],
)
def test_refuses_bad_weights_and_writes_nothing(case, needle, tmp_path, request):
    weights, labels = _weights_case(case, tmp_path, request)
    labels_file = write_labels(tmp_path / "labels.txt", labels)
    out = tmp_path / "out" / "pkg.zip"
    rc, data, raw = run_prepare("--weights", weights, "--labels", labels_file, "--output", out)
    assert rc != 0, raw
    assert data is not None and data.get("result") in ("fail", "error"), raw
    assert needle.lower() in str(data.get("error", "")).lower(), data
    assert not out.exists()


@pytest.mark.parametrize(
    "case, content, needle",
    [
        ("empty", b"", ""),
        ("blank_line", "zebra\n\nñandú\n".encode(), ""),  # 3 lines, one blank
        ("duplicate", b"zebra\nzebra\nokapi\n", "zebra"),
        ("not_utf8", b"zebra\n\xff\xfe\nokapi\n", "utf-8"),
    ],
    ids=["empty", "blank_line", "duplicate", "not_utf8"],
)
def test_refuses_bad_label_files_and_writes_nothing(case, content, needle, r50_file, tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_bytes(content)
    out = tmp_path / "pkg.zip"
    rc, data, raw = run_prepare("--weights", r50_file, "--labels", labels, "--output", out)
    assert rc != 0, raw
    assert data is not None and data.get("result") in ("fail", "error"), raw
    assert needle.lower() in str(data.get("error", "")).lower(), data
    assert not out.exists()


@pytest.mark.parametrize("container", ["raw_pickle", "torch_checkpoint"])
def test_pickle_payload_in_weights_is_never_executed(container, tmp_path):
    marker = tmp_path / "PWNED"
    payload = {"fc.weight": _Plant(marker)}
    if container == "raw_pickle":
        weights = tmp_path / "weights.safetensors"  # the name tempts a naive loader
        weights.write_bytes(pickle.dumps(payload))
    else:
        torch, _timm = _stack()
        weights = tmp_path / "weights.pt"
        torch.save(payload, weights)
    labels = write_labels(tmp_path / "labels.txt", LABELS)
    out = tmp_path / "pkg.zip"
    rc, data, raw = run_prepare("--weights", weights, "--labels", labels, "--output", out)
    assert not marker.exists(), "prepare-finetune unpickled untrusted weights"
    assert rc != 0, raw
    assert data is not None and data.get("result") in ("fail", "error"), raw
    assert not out.exists()


# --------------------------------------------------------------------------- agent toolkit


def test_agent_skill_documents_prepare_finetune():
    """The skill is how an agent finds the command; it must name it and its package schema."""
    skill = Path(__file__).resolve().parents[1] / "skills" / "sparrow-model-uploader"
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(skill.rglob("*.md")))
    assert "prepare-finetune" in text
    assert SCHEMA in text
