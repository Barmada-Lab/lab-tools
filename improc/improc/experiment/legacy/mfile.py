import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import charset_normalizer

@dataclass
class Exposure:
    channel: str
    exposure_ms: int

@dataclass
class DrugInfo:
    drug_label: str
    drug_conc: float | None

@dataclass
class DNAInfo:
    dna_label: str
    dna_conc: float | None

@dataclass
class WellSpec:
    label: str
    exposures: list[Exposure]
    drugs: list[DrugInfo]
    dnas: list[DNAInfo]

@dataclass
class MFSpec:
    name: str
    t_transfect: datetime
    objective: str
    microscope: str
    binning: str
    montage_dim: int
    montage_overlap: float
    imaging_times: list[datetime]
    morphology_channel: str
    wells: list[WellSpec]

def read_mfile(path: pathlib.Path) -> MFSpec | None:
    # There doesn't seem to be a good way to automatically detect character encoding
    # using vanilla python... so we use this
    lines = str(charset_normalizer.from_path(path).best()).split("\n")

    def tokenize(line: str) -> list[str]:
        return [token.strip() for token in line.split(",")]

    def deduplicate(arr) -> list[str]:
        seen = defaultdict(lambda: 0)
        counter = Counter(arr)
        dedup = []
        for x in arr:
            if counter[x] == 1:
                dedup.append(x)
            else:
                count = seen[x]
                dedup.append(f"{x}-{count}")
                seen[x] += 1
        return dedup

    def assoc(fields_str: str, attrs_str: str) -> dict[str, str]:
        fields = deduplicate(tokenize(fields_str))
        attrs = tokenize(attrs_str)
        return dict(zip(fields, attrs))

    gen_spec = assoc(lines[0], lines[1])

    def valid_line(line):
        return not all([token.strip() == "" for token in line.split(",")])

    if not valid_line(lines[3]):
        return None

    well_specs = [assoc(lines[3], line) for line in lines[4:] if valid_line(line)]


    name = gen_spec["PlateID"]
    objective = gen_spec["Objective"]
    microscope = gen_spec["microscope"]
    binning = gen_spec["binning"]
    montage_dim = int(gen_spec["Montage XY"])
    montage_overlap = 1.0 / int(gen_spec["Tile overlap"])
    primary_channel = gen_spec["Primary channel"]

    def parse_crap(date_str: str, time_str: str) -> datetime:
        m, d, y = map(int, date_str.split("/"))
        if time_str in {"", "NA"}:
            hours, mins = (0,0)
        else:
            time_part, ampm = time_str.split(" ")
            hours, mins = map(int, time_part.split(":"))
            if ampm.lower() == "pm" and hours != 12:
                hours += 12
            elif ampm.lower() == "am" and hours == 12:
                hours = 0

        return datetime(hour=hours, minute=mins, month=m, day=d, year=y)

    t_transfect = parse_crap(gen_spec["Transfection date"], gen_spec["Transfection time"])

    def extract_imaging_times(transfection_time: datetime, well_specs: list[dict[str, str]]) -> list[datetime]:
        tps: list[datetime] = []
        for well_spec in well_specs:
            entry = well_spec["Imaging Hours Post-Transfection"]
            try:
                _, hours, _ = entry.split(" ")
                imaging_time = transfection_time + timedelta(hours = int(hours))
                tps.append(imaging_time)
            except ValueError:
                break
        return tps

    def build_wellspec(well_spec: dict[str,str]) -> WellSpec:
        label = well_spec["Well"]
        wells = []
        for idx in range(4):
            fp = well_spec[f"FP{idx+1}"]
            if fp.lower() not in {"cy5", "gfp", "rfp", "dapi", "white_light"}:
                continue
            try:
                exposure = int(well_spec[f"Exposure (ms)-{idx}"])
            except:
                continue
            wells.append(Exposure(fp, exposure))

        drugs = []
        for idx in range(2):
            drug = well_spec[f"Drug{idx+1}"]
            if drug.lower() in ["na","n/a","0"]:
                continue
            drug_conc = well_spec[f"[Drug{idx+1}]"]
            if drug_conc.lower() in ["na", "n/a", "0"]:
                drug_conc = None
            else:
                drug_conc = float(drug_conc)
            drugs.append(DrugInfo(drug, drug_conc))

        dnas = []
        for idx in range(2):
            dna = well_spec[f"DNA{idx+1}"]
            if dna.lower() in ["na","n/a",0]:
                continue
            dna_conc = well_spec[f"[DNA{idx+1}]-ng/well"]
            if dna_conc.lower() in ["na", "n/a", "0"]:
                dna_conc = None
            else:
                dna_conc = float(dna_conc)
            dnas.append(DNAInfo(dna, dna_conc))


        return WellSpec(label, wells, drugs, dnas)

    return MFSpec(
        name=name,
        t_transfect = t_transfect,
        objective = objective,
        microscope = microscope,
        binning = binning,
        montage_dim = montage_dim,
        montage_overlap = montage_overlap,
        morphology_channel=primary_channel,
        imaging_times = extract_imaging_times(t_transfect, well_specs),
        wells = [build_wellspec(well_spec) for well_spec in well_specs]
    )