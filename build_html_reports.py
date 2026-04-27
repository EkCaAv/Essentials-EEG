from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from reporting.html_builder import ReportBuilder


@dataclass
class MinimalConfig:
    experiment_name: str = "pediatric_study"


def discover_file_ids(reports_dir: Path) -> List[str]:
    """
    Extrae file_id desde nombres como:
      chb01_01.edf_psd.png  -> file_id = chb01_01.edf
    """
    rx = re.compile(r"^(?P<fid>.+)_(psd|spectrogram|bandpower_evolution)\.png$")
    fids: Set[str] = set()

    for p in reports_dir.glob("*.png"):
        m = rx.match(p.name)
        if m:
            fids.add(m.group("fid"))

    # También considera spikes/ictal por si faltan globales
    rx2 = re.compile(r"^(?P<fid>.+)_(spikes_.+|ictal\d+_.+)\.png$")
    for p in reports_dir.glob("*.png"):
        m = rx2.match(p.name)
        if m:
            fids.add(m.group("fid"))

    return sorted(fids)


def discover_ictal_indices(reports_dir: Path, file_id: str) -> List[int]:
    rx = re.compile(rf"^{re.escape(file_id)}_ictal(?P<idx>\d+)_")
    idxs: Set[int] = set()
    for p in reports_dir.glob(f"{file_id}_ictal*_*.png"):
        m = rx.match(p.name)
        if m:
            idxs.add(int(m.group("idx")))
    return sorted(idxs)


def build_file_results(reports_dir: Path) -> List[Dict]:
    file_results: List[Dict] = []
    for fid in discover_file_ids(reports_dir):
        ictal_idxs = discover_ictal_indices(reports_dir, fid)
        file_results.append(
            {
                "file": fid,  # importante: debe coincidir con el prefijo de tus pngs
                "has_seizures": len(ictal_idxs) > 0,
                "duration_sec": 0.0,
                "fs": 0.0,
                "n_channels": 0,
                "ictal_summaries": [
                    {"index": idx, "start_s": 0.0, "end_s": 0.0, "duration_s": 0.0}
                    for idx in ictal_idxs
                ],
            }
        )
    return file_results


def main():
    parser = argparse.ArgumentParser(description="Reconstruye reportes HTML desde PNGs existentes")
    parser.add_argument("--results_root", type=str, default="./results_pediatric_study")
    parser.add_argument("--experiment_name", type=str, default="pediatric_study")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    config = MinimalConfig(experiment_name=args.experiment_name)
    builder = ReportBuilder(config)

    if not results_root.exists():
        raise FileNotFoundError(f"No existe results_root: {results_root}")

    subjects = sorted([p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("chb")])

    for subj_dir in subjects:
        reports_dir = subj_dir / "reports"
        if not reports_dir.exists():
            print(f"❌ {subj_dir.name}: no existe carpeta reports/")
            continue

        file_results = build_file_results(reports_dir)
        if not file_results:
            print(f"❌ {subj_dir.name}: no encontré pngs esperados dentro de {reports_dir}")
            continue

        report_path = builder.build(subject_id=subj_dir.name, file_results=file_results, output_dir=reports_dir)
        print(f"✅ {subj_dir.name}: {report_path}")


if __name__ == "__main__":
    main()
