# config/profiles.py
"""
Fase 0 — Cargador del config declarativo único de bandas (band_profiles.yaml).

Fuente de verdad única para las bandas de frecuencia. Tanto el pipeline de
experimentos (pipeline/01_chbmit_experiments.py) como la interfaz (interface/)
consumen este módulo, eliminando la duplicación previa entre
config/base_config.py::BandDefinitions y pipeline::DEFAULT_BANDS.

API principal:
    load_profiles()                  -> dict crudo del YAML
    list_profiles()                  -> [BandProfile, ...]
    get_profile(name)                -> BandProfile
    get_default_profile()            -> BandProfile (línea base de la tesis)
    suggest_profile(age, sex)        -> BandProfile sugerido por demografía
    bands_to_dict(profile)           -> {"delta": (0.5, 4.0), ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Ruta al YAML, relativa a este archivo (robusta ante el cwd).
_CONFIG_PATH = Path(__file__).resolve().parent / "band_profiles.yaml"


@dataclass(frozen=True)
class BandProfile:
    """Un perfil de bandas con su justificación demográfica y científica."""
    name: str
    label: str
    bands: Dict[str, Tuple[float, float]]
    age_min: float
    age_max: float
    sex: str               # "any" | "F" | "M"
    rationale: str
    reference: str
    is_thesis_baseline: bool

    def bands_as_dict(self) -> Dict[str, Tuple[float, float]]:
        """Devuelve las bandas como dict de tuplas (formato del pipeline)."""
        return {k: (float(v[0]), float(v[1])) for k, v in self.bands.items()}

    def applies_to(self, age: Optional[float], sex: Optional[str]) -> bool:
        """True si el perfil aplica a la demografía dada."""
        if age is not None and not (self.age_min <= age <= self.age_max):
            return False
        if sex and self.sex != "any" and self.sex != sex:
            return False
        return True


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------
def load_profiles(path: Path = _CONFIG_PATH) -> dict:
    """Lee y devuelve el contenido crudo del YAML."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el config de perfiles: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "profiles" not in data:
        raise ValueError(f"Config de perfiles inválido (falta 'profiles'): {path}")
    return data


def _parse_profile(name: str, raw: dict) -> BandProfile:
    bands = {k: tuple(v) for k, v in raw["bands"].items()}
    applies = raw.get("applies_to", {}) or {}
    return BandProfile(
        name=name,
        label=raw.get("label", name),
        bands=bands,
        age_min=float(applies.get("age_min", 0)),
        age_max=float(applies.get("age_max", 120)),
        sex=str(applies.get("sex", "any")),
        rationale=" ".join(str(raw.get("rationale", "")).split()),
        reference=str(raw.get("reference", "")).strip(),
        is_thesis_baseline=bool(raw.get("is_thesis_baseline", False)),
    )


def list_profiles(path: Path = _CONFIG_PATH) -> List[BandProfile]:
    """Todos los perfiles definidos, en el orden del YAML."""
    data = load_profiles(path)
    return [_parse_profile(name, raw) for name, raw in data["profiles"].items()]


def get_profile(name: str, path: Path = _CONFIG_PATH) -> BandProfile:
    """Devuelve un perfil por nombre. Lanza KeyError si no existe."""
    data = load_profiles(path)
    profiles = data["profiles"]
    if name not in profiles:
        disponibles = ", ".join(profiles.keys())
        raise KeyError(f"Perfil '{name}' no existe. Disponibles: {disponibles}")
    return _parse_profile(name, profiles[name])


def get_default_profile(path: Path = _CONFIG_PATH) -> BandProfile:
    """Perfil por defecto (línea base reproducible de la tesis)."""
    data = load_profiles(path)
    name = data.get("default_profile")
    if not name or name not in data["profiles"]:
        # Fallback: el primero marcado como baseline, o el primero a secas.
        for n, raw in data["profiles"].items():
            if raw.get("is_thesis_baseline"):
                name = n
                break
        else:
            name = next(iter(data["profiles"]))
    return _parse_profile(name, data["profiles"][name])


def suggest_profile(age: Optional[float] = None,
                    sex: Optional[str] = None,
                    path: Path = _CONFIG_PATH) -> BandProfile:
    """
    Sugiere el perfil más específico que aplica a la demografía dada.
    'Más específico' = menor rango de edad cubierto. Si ninguno aplica,
    devuelve el perfil por defecto.
    """
    candidatos = [p for p in list_profiles(path) if p.applies_to(age, sex)]
    if not candidatos:
        return get_default_profile(path)
    # El de rango etario más estrecho es el más específico.
    return min(candidatos, key=lambda p: p.age_max - p.age_min)


def bands_to_dict(profile: BandProfile) -> Dict[str, Tuple[float, float]]:
    """Atajo: bandas de un perfil como dict de tuplas."""
    return profile.bands_as_dict()


if __name__ == "__main__":
    # Smoke test manual: python -m config.profiles
    print(f"Config: {_CONFIG_PATH}")
    print(f"Default: {get_default_profile().name}\n")
    for p in list_profiles():
        flag = "  [BASELINE TESIS]" if p.is_thesis_baseline else ""
        print(f"- {p.name}: {p.label}{flag}")
        print(f"    edad {p.age_min:.0f}-{p.age_max:.0f} | sexo {p.sex}")
        print(f"    bandas: {p.bands_as_dict()}")
    print(f"\nSugerencia para edad=7: {suggest_profile(age=7).name}")
    print(f"Sugerencia para edad=25: {suggest_profile(age=25).name}")
