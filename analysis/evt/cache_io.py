from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from config import config_fingerprint, config_snapshot

ArrayT = TypeVar("ArrayT", bound=np.ndarray)


class ConfigFingerprintMismatch(RuntimeError):
    pass


def fingerprint_sidecar_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(artifact_path.name + ".fingerprint.json")


def write_fingerprint(artifact_path: Path, extra: dict[str, Any] | None = None):
    payload: dict[str, Any] = {
        "fingerprint": config_fingerprint(),
        "config": config_snapshot(),
        "artifact": artifact_path.name,
    }
    if extra:
        payload.update(extra)
    sidecar = fingerprint_sidecar_path(artifact_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")


def read_fingerprint_record(artifact_path: Path) -> dict[str, Any] | None:
    sidecar = fingerprint_sidecar_path(artifact_path)
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text())


def assert_fingerprint(artifact_path: Path) -> str:
    record = read_fingerprint_record(artifact_path)
    expected = config_fingerprint()
    if record is None:
        raise ConfigFingerprintMismatch(
            f"{artifact_path.name}: missing fingerprint sidecar "
            f"(current config fingerprint={expected})"
        )
    found = str(record.get("fingerprint", ""))
    if found != expected:
        raise ConfigFingerprintMismatch(
            f"{artifact_path.name}: fingerprint mismatch "
            f"cached={found} current={expected}"
        )
    return found


def save_array(
    path: Path,
    array: np.ndarray,
    *,
    extra: dict[str, Any] | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    write_fingerprint(path, extra=extra)


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    assert_fingerprint(path)
    return np.load(path)


def load_or_compute_array(
    path: Path,
    compute: Callable[..., ArrayT],
    *args: object,
    **kwargs: object,
) -> ArrayT:
    if path.exists():
        try:
            return load_array(path)
        except ConfigFingerprintMismatch:
            path.unlink(missing_ok=True)
            fingerprint_sidecar_path(path).unlink(missing_ok=True)
    array = compute(*args, **kwargs)
    save_array(path, array)
    return array


def save_json(path: Path, payload: dict[str, Any]):
    stamped = dict(payload)
    stamped["config_fingerprint"] = config_fingerprint()
    stamped["config"] = config_snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamped, indent=2) + "\n")
    write_fingerprint(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    expected = config_fingerprint()
    found = payload.get("config_fingerprint")
    if found is None:
        sidecar = read_fingerprint_record(path)
        found = None if sidecar is None else sidecar.get("fingerprint")
    if found is None:
        raise ConfigFingerprintMismatch(
            f"{path.name}: missing config_fingerprint "
            f"(current={expected})"
        )
    if found != expected:
        raise ConfigFingerprintMismatch(
            f"{path.name}: fingerprint mismatch cached={found} current={expected}"
        )
    return payload


def inspect_artifact_fingerprint(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "current_fingerprint": config_fingerprint(),
    }
    if not path.exists():
        status["status"] = "missing"
        return status
    record = read_fingerprint_record(path)
    if record is None and path.suffix == ".json":
        try:
            payload = json.loads(path.read_text())
            if "config_fingerprint" in payload:
                record = {
                    "fingerprint": payload["config_fingerprint"],
                    "config": payload.get("config"),
                }
        except (OSError, json.JSONDecodeError):
            record = None
    if record is None:
        status["status"] = "no_fingerprint"
        status["cached_fingerprint"] = None
        return status
    cached = str(record.get("fingerprint", ""))
    status["cached_fingerprint"] = cached
    status["status"] = "match" if cached == config_fingerprint() else "mismatch"
    if "config" in record and isinstance(record["config"], dict):
        status["cached_highpass_hz"] = record["config"].get("HIGHPASS_CUTOFF_HZ")
        status["cached_ar_phi"] = record["config"].get("AR_PHI")
        status["cached_burst_subtle"] = record["config"].get("BURST_STD_SUBTLE")
    return status
