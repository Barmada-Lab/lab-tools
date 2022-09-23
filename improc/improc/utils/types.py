from dataclasses import dataclass

from datetime import datetime

@dataclass
class Exposure:
    channel: str
    exposure_ms: int

@dataclass
class DrugInfo:
    drug_label: str
    drug_conc: float | None

@dataclass
class WellSpec:
    label: str
    exposures: list[Exposure]
    drugs: list[DrugInfo]

@dataclass
class MFSpec:
    name: str
    t_transfect: datetime
    objective: str
    microscope: str
    binning: str
    montage_dim: int
    montage_overlap: float
    wells: list[WellSpec]
