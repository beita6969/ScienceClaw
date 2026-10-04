"""The 23 ScienceClaw-Eval disciplines (ANZSRC 2020 FoR divisions 30–52), families and adapters."""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

DATA_ROOT = os.environ.get("SCIENCECLAW_DATA_ROOT", "/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets")

FAMILIES = ("Life & health", "Physical & Earth", "Engineering & computing", "Social & behavior", "Humanities & law")


@dataclass(frozen=True)
class Discipline:
    code: str          # "FoR34"
    name: str          # ANZSRC division name
    family: str
    dataset: str       # recorded IID source dataset
    metric: str
    direction: str     # "max" | "min"
    module: str        # adapter module under scienceclaw.bench.tasks


DISCIPLINES: tuple[Discipline, ...] = (
    Discipline("FoR30", "Agricultural, veterinary and food sciences", "Life & health", "PhenoBench hierarchical panoptic segmentation", "PQ+", "max", "for30_phenobench"),
    Discipline("FoR31", "Biological sciences", "Life & health", "ProteinGym substitutions", "mean Spearman", "max", "for31_proteingym"),
    Discipline("FoR32", "Biomedical and clinical sciences", "Life & health", "MSD Task04 Hippocampus", "DSC", "max", "for32_msd_hippocampus"),
    Discipline("FoR33", "Built environment and design", "Engineering & computing", "BuildingsBench", "balanced NRMSE (%)", "min", "for33_buildingsbench"),
    Discipline("FoR34", "Chemical sciences", "Physical & Earth", "OGB ogbg-molhiv", "ROC-AUC", "max", "for34_molhiv"),
    Discipline("FoR35", "Commerce, management, tourism and services", "Social & behavior", "Monash Tourism Monthly", "mean MASE", "min", "for35_tourism"),
    Discipline("FoR36", "Creative arts and writing", "Humanities & law", "MUSDB18 four-stem separation", "mean target-median SDR (dB)", "max", "for36_musdb"),
    Discipline("FoR37", "Earth sciences", "Physical & Earth", "WeatherBench2", "2m temperature RMSE (K)", "min", "for37_weatherbench"),
    Discipline("FoR38", "Economics", "Social & behavior", "World Bank macro forecasting", "mean sMAPE (%)", "min", "for38_worldbank"),
    Discipline("FoR39", "Education", "Social & behavior", "Eedi NeurIPS 2020 Task 4", "organizer 10-mask accuracy", "max", "for39_eedi"),
    Discipline("FoR40", "Engineering", "Engineering & computing", "DCASE 2024 Task 2", "official DCASE score", "max", "for40_dcase"),
    Discipline("FoR41", "Environmental sciences", "Physical & Earth", "NEON aquatics forecasting", "mean CRPS (oxygen + temperature)", "min", "for41_neon"),
    Discipline("FoR42", "Health sciences", "Life & health", "PhysioNet/CinC 2019 sepsis", "normalized clinical utility", "max", "for42_sepsis"),
    Discipline("FoR43", "History, heritage and archaeology", "Humanities & law", "HIPE-OCRepair 2026", "weighted cMER-micro", "min", "for43_hipe"),
    Discipline("FoR44", "Human society", "Social & behavior", "ACIC 2016", "response-SD normalized RMSE", "min", "for44_acic"),
    Discipline("FoR45", "Indigenous studies", "Humanities & law", "AmericasNLP 2026", "mean sentence chrF++", "max", "for45_americasnlp"),
    Discipline("FoR46", "Information and computing sciences", "Engineering & computing", "HumanEval → MBPP", "execution pass@1", "max", "for46_code"),
    Discipline("FoR47", "Language, communication and culture", "Humanities & law", "CoNLL-2018 UD", "LAS", "max", "for47_ud"),
    Discipline("FoR48", "Law and legal studies", "Humanities & law", "ContractNLI", "mAP", "max", "for48_contractnli"),
    Discipline("FoR49", "Mathematical sciences", "Engineering & computing", "SMT-COMP 2025 QF_NIA", "oracle-agreement accuracy", "max", "for49_smt"),
    Discipline("FoR50", "Philosophy and religious studies", "Humanities & law", "SemEval-2023 Task 4 ValueEval", "official F1", "max", "for50_valueeval"),
    Discipline("FoR51", "Physical sciences", "Physical & Earth", "Matbench phonons", "MAE", "min", "for51_matbench"),
    Discipline("FoR52", "Psychology", "Social & behavior", "Psych-201 discrete", "micro accuracy", "max", "for52_psych201"),
)

BY_CODE = {d.code: d for d in DISCIPLINES}


def family_of(code: str) -> str:
    return BY_CODE[code].family


def get_adapter(code: str, **kwargs):
    """Instantiate the adapter of a discipline (module must define `Adapter`)."""
    d = BY_CODE[code]
    mod = importlib.import_module(f"scienceclaw.bench.tasks.{d.module}")
    return mod.Adapter(**kwargs)


def available_adapters(**kwargs) -> dict[str, object]:
    out = {}
    for d in DISCIPLINES:
        try:
            a = get_adapter(d.code, **kwargs)
        except ModuleNotFoundError:
            continue
        ok, _ = a.available()
        if ok:
            out[d.code] = a
    return out
