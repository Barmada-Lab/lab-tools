
from lifelines import CoxPHFitter
import pandas as pd

from improc.experiment.types import Experiment

def analyze(experiment: Experiment):
    surv_data_path = experiment.experiment_dir / "results" / "survival.csv"
    surv_data = pd.read_csv(surv_data_path)

    surv_data["well"] = surv_data["well"].str.slice(0, 3)
    for well in experiment.mfspec.wells:
        group = "-".join([dna.dna_label for dna in well.dnas] + [drug.drug_label for drug in well.drugs])
        surv_data.loc[surv_data["well"] == well.label, "group"] = group

    df = surv_data.drop(["id","well"], axis=1)
    cph = CoxPHFitter()
    cph.fit(df, duration_col="tp", event_col="event", strata="group")
    return cph