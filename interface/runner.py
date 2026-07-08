# interface/runner.py
"""
Lanzamiento del pipeline de experimentos como proceso en segundo plano.

Una corrida completa (112 entrenamientos) tarda 1–3 h, así que NO puede ser
síncrona dentro de la app. Aquí se construye el comando, se lanza como
subproceso desacoplado y se siguen sus logs sin bloquear la interfaz.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = REPO_ROOT / "pipeline" / "01_chbmit_experiments.py"
RUNS_DIR = REPO_ROOT / "interface" / "_runs"  # logs de corridas lanzadas desde la UI


@dataclass
class RunRequest:
    """Parámetros de una corrida solicitada desde la interfaz."""
    subjects: List[str]
    band_profile: str
    out_dir: str
    run_name: str
    window_sec: float = 5.0
    overlap: float = 0.5
    n_splits: int = 7
    resample_hz: float = 256.0
    max_interictal_per_file: int = 300
    extra_args: List[str] = field(default_factory=list)

    def to_command(self) -> List[str]:
        cmd = [
            sys.executable, str(PIPELINE_SCRIPT),
            "--data_root", str(REPO_ROOT / "data"),
            "--out_dir", self.out_dir,
            "--subjects", *self.subjects,
            "--run_name", self.run_name,
            "--band_profile", self.band_profile,
            "--window_sec", str(self.window_sec),
            "--overlap", str(self.overlap),
            "--n_splits", str(self.n_splits),
            "--resample_hz", str(self.resample_hz),
            "--max_interictal_per_file", str(self.max_interictal_per_file),
            *self.extra_args,
        ]
        return cmd


@dataclass
class LaunchedRun:
    run_name: str
    pid: int
    log_path: Path
    command: List[str]
    started_at: str


def launch(req: RunRequest) -> LaunchedRun:
    """
    Lanza el pipeline en background. Devuelve metadatos (PID + ruta de log).
    El proceso sobrevive aunque la pestaña del navegador se cierre.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RUNS_DIR / f"{req.run_name}_{stamp}.log"

    cmd = req.to_command()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# Comando: {' '.join(cmd)}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    return LaunchedRun(
        run_name=req.run_name,
        pid=proc.pid,
        log_path=log_path,
        command=cmd,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )


def read_log_tail(log_path: Path, max_lines: int = 60) -> str:
    """Devuelve las últimas líneas del log de una corrida."""
    if not log_path.exists():
        return "(sin log todavía)"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return "(no se pudo leer el log)"
    return "\n".join(lines[-max_lines:])


def is_running(pid: int) -> bool:
    """Comprueba si un PID sigue vivo (multiplataforma, best-effort)."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True,
            )
            return str(pid) in out.stdout
        import os
        os.kill(pid, 0)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
