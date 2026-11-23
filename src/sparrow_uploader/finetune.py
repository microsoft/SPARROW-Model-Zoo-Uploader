"""``prepare-finetune``: build a local training package for the Sparrow fine-tune service.

The package (``schema=sparrow-finetune/base-v1``) is a ZIP_STORED archive with exactly
``manifest.json`` and ``weights.safetensors``. The service imports it with
``POST /v1/base-models/local`` and warm-starts ``conventional_classifier`` jobs from it.

The architecture is declared by the user (``--model-arch``, default ``resnet50``) and must be one
of ``ARCHITECTURES``: timm (PyTorch Image Models) can build many networks, but only these have
been validated end to end with the service. It is never inferred from the weights.

Nothing here touches the network or a workspace. The weights file is parsed as safetensors only
(never unpickled), matched strictly against the declared timm architecture, and one real CPU
training step is run on a copy to prove the tensors train. The package holds the input tensors
unchanged.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .workspace import GateFailed, UploaderError

SCHEMA = "sparrow-finetune/base-v1"
PIPELINE = "conventional_classifier"
MODEL_ARCH = "resnet50"  # the default architecture
# Validated timm architecture -> its classifier head weight (one row per label).
ARCHITECTURES = {"resnet50": "fc.weight", "efficientnet_b0": "classifier.weight"}
MANIFEST = "manifest.json"
WEIGHTS = "weights.safetensors"

# Bound metadata separately from model tensors.
MAX_WEIGHTS_BYTES = 1 << 30
MAX_MANIFEST_BYTES = 16 << 20
MAX_HEADER_BYTES = 16 << 20
MAX_LABELS = 100_000
MAX_LABEL_CHARS = 256
# With one class, cross-entropy is constant and the head cannot learn anything.
MIN_LABELS = 2
MAX_LABELS_BYTES = 4 << 20

# safetensors dtype -> element size in bytes.
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
_INSTALL_HINT = (
    "install the fine-tune extra: uv tool install "
    "'sparrow-model-uploader[finetune] @ git+https://github.com/microsoft/SPARROW-Model-Zoo-Uploader'"
)


def _stack():
    try:
        import timm
        import torch
        from safetensors.torch import load as st_load
        from safetensors.torch import save as st_save
    except ImportError as exc:
        raise UploaderError(f"prepare-finetune needs torch, timm and safetensors ({exc}); {_INSTALL_HINT}") from exc
    return torch, timm, st_load, st_save


# --------------------------------------------------------------------------- inputs


def read_labels(path: Path) -> list[str]:
    """One label per line, UTF-8, order kept. Empty, blank, duplicate or odd labels are refused."""
    if not path.is_file():
        raise UploaderError(f"labels file not found: {path}")
    if path.stat().st_size > MAX_LABELS_BYTES:
        raise UploaderError(f"labels file is larger than {MAX_LABELS_BYTES} bytes: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploaderError(f"labels file is not UTF-8 text ({exc}): {path}") from exc
    text = text.removeprefix("\ufeff")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # the trailing newline
    labels = [line.removesuffix("\r") for line in lines]
    if not labels:
        raise UploaderError(f"labels file is empty: {path}")
    if len(labels) < MIN_LABELS:
        raise UploaderError(
            f"labels file has a single label; a classifier needs at least {MIN_LABELS} "
            "(with one class there is nothing to tell apart, so it cannot be trained)"
        )
    if len(labels) > MAX_LABELS:
        raise UploaderError(f"labels file has {len(labels)} labels; the limit is {MAX_LABELS}")
    seen: dict[str, int] = {}
    for n, label in enumerate(labels, 1):
        if not label.strip():
            raise UploaderError(f"labels file line {n} is blank; write one label per line, no blank lines")
        if label != label.strip():
            raise UploaderError(f"labels file line {n} has leading or trailing whitespace: {label!r}")
        if len(label) > MAX_LABEL_CHARS:
            raise UploaderError(f"labels file line {n} is longer than {MAX_LABEL_CHARS} characters")
        if any(ord(c) < 32 or ord(c) == 127 for c in label):
            raise UploaderError(f"labels file line {n} holds a control character: {label!r}")
        if label in seen:
            raise UploaderError(f"labels file repeats label {label!r} (lines {seen[label]} and {n})")
        seen[label] = n
    return labels


def check_safetensors(data: bytes) -> dict[str, dict]:
    """Validate the safetensors framing without any loader; returns the tensor header entries.

    A pickle or torch checkpoint fails here on its first bytes, so it is never deserialised."""
    if len(data) < 8:
        raise GateFailed("weights file is not safetensors: shorter than the 8-byte header length")
    n = int.from_bytes(data[:8], "little")
    if n < 2 or n > MAX_HEADER_BYTES or 8 + n > len(data):
        raise GateFailed(
            f"weights file is not safetensors: header length {n} does not fit a {len(data)}-byte file "
            "(pickle and .pt/.pth checkpoints are not accepted; save the state dict with safetensors)"
        )
    try:
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise GateFailed(f"weights file is not safetensors: header is not JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise GateFailed("weights file is not safetensors: header is not a JSON object")
    meta = header.get("__metadata__")
    if meta is not None and not (isinstance(meta, dict) and all(isinstance(v, str) for v in meta.values())):
        raise GateFailed("weights file is not valid safetensors: __metadata__ must map strings to strings")
    buffer = len(data) - 8 - n
    tensors: dict[str, dict] = {}
    spans = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        # Type-check every nested value before using it: a list or object where a string or number
        # belongs is a refusal, never a crash (an unhashable dtype would raise TypeError).
        ok = (
            isinstance(entry, dict)
            and isinstance(entry.get("dtype"), str)
            and entry["dtype"] in _DTYPE_BYTES
            and isinstance(entry.get("shape"), list)
            and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in entry["shape"])
            and isinstance(entry.get("data_offsets"), list)
            and len(entry["data_offsets"]) == 2
            and all(isinstance(o, int) and not isinstance(o, bool) for o in entry["data_offsets"])
        )
        if not ok:
            raise GateFailed(f"weights file is not valid safetensors: bad header entry for {name!r}")
        begin, end = entry["data_offsets"]
        size = math.prod(entry["shape"]) * _DTYPE_BYTES[entry["dtype"]]
        if not 0 <= begin <= end <= buffer or end - begin != size:
            raise GateFailed(f"weights file is not valid safetensors: tensor {name!r} lies outside the data")
        spans.append((begin, end))
        tensors[name] = entry
    spans.sort()
    pos = 0
    for begin, end in spans:
        if begin != pos:
            raise GateFailed("weights file is not valid safetensors: tensor data overlaps or has gaps")
        pos = end
    if pos != buffer:
        raise GateFailed("weights file is not valid safetensors: trailing bytes after the tensor data")
    if not tensors:
        raise GateFailed("weights file is safetensors but holds no tensors")
    return tensors


def _names(items, limit: int = 8) -> str:
    items = sorted(items)
    more = f" and {len(items) - limit} more" if len(items) > limit else ""
    return ", ".join(items[:limit]) + more


def check_model_arch(model_arch: str) -> str:
    """Only validated architectures; a name timm happens to know is not enough."""
    if model_arch not in ARCHITECTURES:
        raise UploaderError(
            f"model architecture {model_arch!r} is not supported; choose one of "
            f"{', '.join(ARCHITECTURES)}. timm (PyTorch Image Models) can build other networks, "
            "but they have not been validated with the fine-tune service"
        )
    return model_arch


def check_state(state: dict, labels: list[str], torch, timm, model_arch: str = MODEL_ARCH) -> Any:
    """Strict match against timm ``model_arch`` with a head of ``len(labels)`` rows; returns the model."""
    head_key = ARCHITECTURES[check_model_arch(model_arch)]
    model = timm.create_model(model_arch, pretrained=False, num_classes=len(labels))
    ref = model.state_dict()
    missing, extra = set(ref) - set(state), set(state) - set(ref)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing tensors {_names(missing)}")
        if extra:
            parts.append(f"unexpected tensors {_names(extra)}")
        raise GateFailed(
            f"weights are not a timm {model_arch} state dict: " + "; ".join(parts)
            + f" (the architecture is the one declared with --model-arch, one of {', '.join(ARCHITECTURES)};"
            f" the default is {MODEL_ARCH})"
        )
    head, ref_head = state[head_key], ref[head_key]
    if head.dim() == 2 and head.shape[1] == ref_head.shape[1] and head.shape[0] != len(labels):
        raise GateFailed(
            f"the {model_arch} classifier head ({head_key.removesuffix('.weight')}) has {head.shape[0]} "
            f"outputs but the labels file has {len(labels)} labels; they must match one label per head "
            "row, in head order"
        )
    shapes = [
        f"{k} {tuple(v.shape)} (expected {tuple(ref[k].shape)})"
        for k, v in state.items() if tuple(v.shape) != tuple(ref[k].shape)
    ]
    if shapes:
        raise GateFailed(f"weights do not have timm {model_arch} shapes: " + "; ".join(sorted(shapes)[:8]))
    dtypes = [f"{k} {v.dtype} (expected {ref[k].dtype})" for k, v in state.items() if v.dtype != ref[k].dtype]
    if dtypes:
        raise GateFailed(f"weights do not have timm {model_arch} dtypes: " + "; ".join(sorted(dtypes)[:8]))
    bad = [k for k, v in state.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
    if bad:
        raise GateFailed(f"weights hold NaN or infinite values in {_names(bad)}; they cannot train")
    return model


def check_resnet50(state: dict, labels: list[str], torch, timm) -> Any:
    """Earlier name of ``check_state`` for the default architecture; kept for existing callers."""
    return check_state(state, labels, torch, timm, MODEL_ARCH)


def training_step(model, state: dict, n_labels: int, torch) -> dict:
    """One forward/backward/SGD step on CPU with the weights loaded into ``model`` (a copy).

    Proves the tensors load strictly and train: finite loss, finite non-zero gradients, finite
    parameters after the update."""
    model.load_state_dict(state, strict=True)  # copies; ``state`` is not modified
    model.to("cpu").train()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 128, 128, generator=gen)
    y = torch.arange(2) % n_labels
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    if not bool(torch.isfinite(loss)):
        raise GateFailed(f"the CPU training step gave a non-finite loss ({loss.item()}); the weights cannot train")
    loss.backward()
    norms = []
    for name, p in model.named_parameters():
        if p.grad is None:
            raise GateFailed(f"the CPU training step produced no gradient for {name}")
        if not bool(torch.isfinite(p.grad).all()):
            raise GateFailed(f"the CPU training step produced non-finite gradients for {name}")
        norms.append(p.grad.detach().pow(2).sum())
    grad_norm = float(torch.stack(norms).sum().sqrt())
    if grad_norm == 0.0:
        raise GateFailed("the CPU training step produced all-zero gradients; the weights cannot train")
    opt.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise GateFailed("the CPU training step produced non-finite weights after one update")
    return {"device": "cpu", "steps": 1, "batch": [2, 3, 128, 128], "loss": loss.item(), "grad_norm": grad_norm}


# --------------------------------------------------------------------------- output


def _zip_bytes(manifest: bytes, weights: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in ((MANIFEST, manifest), (WEIGHTS, weights)):
            info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return buf.getvalue()


def _check_zip(data: bytes, manifest: dict) -> None:
    """Re-read the archive as the service will: exact members, stored, hash matches."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
        if sorted(i.filename for i in infos) != sorted([MANIFEST, WEIGHTS]) or len(infos) != 2:
            raise UploaderError("internal error: package members are wrong")
        if any(i.compress_type != zipfile.ZIP_STORED for i in infos) or zf.testzip() is not None:
            raise UploaderError("internal error: package members are not stored intact")
        if json.loads(zf.read(MANIFEST).decode("utf-8")) != manifest:
            raise UploaderError("internal error: package manifest does not round-trip")
        if hashlib.sha256(zf.read(WEIGHTS)).hexdigest() != manifest["files"][WEIGHTS]:
            raise UploaderError("internal error: package weights do not match the manifest hash")


def _write_atomic(out: Path, data: bytes, force: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if force:
            os.replace(tmp, out)
        else:
            try:
                os.link(tmp, out)  # fails if ``out`` appeared meanwhile: never clobber
            except FileExistsError:
                raise UploaderError(f"output already exists: {out} (pass --force to replace it)") from None
            except OSError:
                if out.exists():
                    raise UploaderError(f"output already exists: {out} (pass --force to replace it)") from None
                os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def prepare_finetune(
    weights: Path, labels: Path, output: Path, force: bool = False, model_arch: str = MODEL_ARCH
) -> dict:
    check_model_arch(model_arch)
    weights, labels, output = Path(weights), Path(labels), Path(output)
    if not weights.is_file():
        raise UploaderError(f"weights file not found: {weights}")
    if output.exists() and not force:
        raise UploaderError(f"output already exists: {output} (pass --force to replace it)")
    if output.exists() and not output.is_file():
        raise UploaderError(f"output is not a file: {output}")
    resolved = output.resolve()
    if resolved in (weights.resolve(), labels.resolve()):
        raise UploaderError("--output must not be the weights or labels file")
    names = read_labels(labels)
    size = weights.stat().st_size
    if size > MAX_WEIGHTS_BYTES:
        raise GateFailed(f"weights file is {size} bytes; the fine-tune service accepts at most {MAX_WEIGHTS_BYTES}")
    data = weights.read_bytes()
    check_safetensors(data)

    torch, timm, st_load, st_save = _stack()
    try:
        state = st_load(data)
    except Exception as exc:  # noqa: BLE001 - the safetensors loader raises its own error types
        raise GateFailed(f"weights file is not valid safetensors: {exc}") from exc
    del data
    model = check_state(state, names, torch, timm, model_arch)
    # Serialise the input tensors before training: the package can only hold them unchanged.
    # Re-serialising also drops any safetensors __metadata__ (free text that may hold paths).
    payload = st_save(state)
    check = training_step(model, state, len(names), torch)

    manifest = {
        "schema": SCHEMA,
        "pipeline": PIPELINE,
        "model_arch": model_arch,
        "labels": names,
        "files": {WEIGHTS: hashlib.sha256(payload).hexdigest()},
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise GateFailed(
            f"the manifest would be {len(manifest_bytes)} bytes; the fine-tune service accepts at most "
            f"{MAX_MANIFEST_BYTES}"
        )
    archive = _zip_bytes(manifest_bytes, payload)
    _check_zip(archive, manifest)
    _write_atomic(output, archive, force)
    return {
        "result": "pass",
        "package": str(output),
        "sha256": hashlib.sha256(archive).hexdigest(),
        "size_bytes": len(archive),
        "schema": SCHEMA,
        "pipeline": PIPELINE,
        "model_arch": model_arch,
        "labels": len(names),
        "tensors": len(state),
        "weights_sha256": manifest["files"][WEIGHTS],
        "training_check": check,
        "next": "nothing has left this machine and nothing was published to the model zoo; to train "
        "from it, import this file into the Sparrow fine-tune service as a local base model "
        "(POST /v1/base-models/local), where other users of that service can see it",
    }
