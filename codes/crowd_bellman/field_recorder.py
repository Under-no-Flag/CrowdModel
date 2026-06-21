from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .metrics import save_json
from .scenes import SimulationConfig


StepObserver = Callable[[dict[str, object]], None]


@dataclass
class FieldDataRecorder:
    output_dir: Path
    save_every: int
    dtype: str = "float32"
    fields: tuple[str, ...] = ("rho", "speed", "vx", "vy")
    _files: list[dict[str, object]] = field(default_factory=list)
    _shape: tuple[int, ...] | None = None
    _static_masks_file: str | None = None

    def __post_init__(self) -> None:
        if self.save_every <= 0:
            raise ValueError("save_every must be positive")
        np.dtype(self.dtype)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def observe(self, snapshot: dict[str, object]) -> None:
        step = int(snapshot["step"])
        is_final = bool(snapshot.get("is_final", False))
        if (step % self.save_every) != 0 and not is_final:
            return

        self._write_static_masks(snapshot)

        payload: dict[str, np.ndarray] = {
            "step": np.array([step], dtype=np.int64),
            "time": np.array([float(snapshot["time"])], dtype=np.float64),
            "dt": np.array([float(snapshot["dt"])], dtype=np.float64),
        }
        for field_name in self.fields:
            payload[field_name] = np.asarray(snapshot[field_name], dtype=self.dtype)

        shape = payload[self.fields[0]].shape
        if self._shape is None:
            self._shape = tuple(int(value) for value in shape)
        elif tuple(shape) != self._shape:
            raise ValueError(f"Field shape changed from {self._shape} to {tuple(shape)}")

        filename = f"field_step_{step:04d}.npz"
        path = self.output_dir / filename
        np.savez_compressed(path, **payload)
        self._files.append(
            {
                "file": filename,
                "step": step,
                "time": float(snapshot["time"]),
                "dt": float(snapshot["dt"]),
                "is_final": is_final,
            }
        )
        self._write_manifest()

    def _write_static_masks(self, snapshot: dict[str, object]) -> None:
        if self._static_masks_file is not None or "walkable" not in snapshot:
            return
        filename = "static_masks.npz"
        path = self.output_dir / filename
        np.savez_compressed(path, walkable=np.asarray(snapshot["walkable"], dtype=bool))
        self._static_masks_file = filename

    def _write_manifest(self) -> None:
        payload: dict[str, object] = {
            "format": "npz_compressed",
            "save_every": int(self.save_every),
            "dtype": str(np.dtype(self.dtype).name),
            "fields": list(self.fields),
            "shape": list(self._shape or ()),
            "files": self._files,
        }
        if self._static_masks_file is not None:
            payload["static_masks"] = self._static_masks_file
        save_json(self.output_dir / "fields_manifest.json", payload)


def make_field_data_observer_factory(
    *,
    save_every: int | None = None,
    dtype: str = "float32",
    output_dir_name: str = "fields",
) -> Callable[..., StepObserver]:
    np.dtype(dtype)
    if save_every is not None and save_every <= 0:
        raise ValueError("save_every must be positive")

    def factory(*, case_output_dir: Path, simulation: SimulationConfig, **_: object) -> StepObserver:
        interval = int(save_every if save_every is not None else simulation.save_every)
        recorder = FieldDataRecorder(
            output_dir=case_output_dir / output_dir_name,
            save_every=interval,
            dtype=dtype,
        )
        return recorder.observe

    return factory
