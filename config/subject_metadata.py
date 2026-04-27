# config/subject_metadata.py
"""
Base de datos de metadatos clínicos de los sujetos CHB-MIT.
Información extraída de la documentación oficial del dataset.
"""

SUBJECTS_DB = {
    # === GRUPO PEDIÁTRICO (≤11 años) ===
    "chb01": {"age": 11, "gender": "F", "group": "pediatric"},
    "chb02": {"age": 11, "gender": "M", "group": "pediatric"},
    "chb03": {"age": 14, "gender": "F", "group": "adolescent"},
    "chb04": {"age": 22, "gender": "M", "group": "adult"},
    "chb05": {"age": 7,  "gender": "F", "group": "pediatric"},
    "chb06": {"age": 1,  "gender": "F", "group": "pediatric"},
    "chb07": {"age": 14, "gender": "F", "group": "adolescent"},
    "chb08": {"age": 3,  "gender": "M", "group": "pediatric"},
    "chb09": {"age": 10, "gender": "F", "group": "pediatric"},
    "chb10": {"age": 3,  "gender": "M", "group": "pediatric"},
    "chb11": {"age": 12, "gender": "F", "group": "adolescent"},
    "chb12": {"age": 2,  "gender": "F", "group": "pediatric"},
    "chb13": {"age": 3,  "gender": "F", "group": "pediatric"},
    "chb14": {"age": 9,  "gender": "F", "group": "pediatric"},
    "chb15": {"age": 16, "gender": "M", "group": "adolescent"},
    "chb16": {"age": 7,  "gender": "F", "group": "pediatric"},
    "chb17": {"age": 12, "gender": "F", "group": "adolescent"},
    "chb18": {"age": 18, "gender": "F", "group": "adult"},
    "chb19": {"age": 19, "gender": "F", "group": "adult"},
    "chb20": {"age": 6,  "gender": "F", "group": "pediatric"},
    "chb21": {"age": 13, "gender": "F", "group": "adolescent"},
    "chb22": {"age": 9,  "gender": "F", "group": "pediatric"},
    "chb23": {"age": 6,  "gender": "F", "group": "pediatric"},
    "chb24": {"age": 16, "gender": "N/A", "group": "adolescent"},
}


def get_subjects(max_age: float = None, min_age: float = None, group: str = None):
    """
    Obtiene lista de sujetos con filtros opcionales.
    
    Args:
        max_age: Edad máxima (inclusive con <)
        min_age: Edad mínima (inclusive)
        group: Filtrar por grupo ('pediatric', 'adolescent', 'adult')
    
    Returns:
        Lista ordenada de subject_ids
    """
    subjects = list(SUBJECTS_DB.keys())
    
    if max_age is not None:
        subjects = [s for s in subjects if SUBJECTS_DB[s]["age"] < max_age]
    
    if min_age is not None:
        subjects = [s for s in subjects if SUBJECTS_DB[s]["age"] >= min_age]
    
    if group is not None:
        subjects = [s for s in subjects if SUBJECTS_DB[s]["group"] == group]
    
    return sorted(subjects)


def get_pediatric_subjects(max_age: int = 11):
    """Retorna lista de IDs de sujetos pediátricos estrictos (≤11 años)"""
    return [sid for sid, meta in SUBJECTS_DB.items() if meta["age"] <= max_age]


def get_subjects_by_group(group: str):
    """Retorna lista de IDs por grupo (pediatric, adolescent, adult)"""
    return [sid for sid, meta in SUBJECTS_DB.items() if meta["group"] == group]


def get_subject_info(subject_id: str) -> dict:
    """Retorna metadatos de un sujeto específico"""
    return SUBJECTS_DB.get(subject_id, {"age": None, "gender": None, "group": None})