import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
R = "/Users/admin/Datasets/ScienceClaw-rebuild-20260928/datasets/for34-ogbg-molhiv/extracted/hiv/"
m = pd.read_csv(R + "mapping/mol.csv.gz")
sp = {k: pd.read_csv(R + f"split/scaffold/{k}.csv.gz", header=None)[0].values for k in ("train", "valid", "test")}
print({k: len(v) for k, v in sp.items()}, "cols", list(m.columns))
def canon(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        return s if mol is None else Chem.MolToSmiles(mol)
    return Chem.MolToSmiles(mol)
def scaf(s):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
    except Exception:
        return None
can = m["smiles"].map(canon).values
sc = m["smiles"].map(scaf).values
lab = m["HIV_active"].values
for ev in ("valid", "test"):
    tr = set(can[sp["train"]]); dup = [i for i in sp[ev] if can[i] in tr]
    trs = set(sc[sp["train"]]); sdup = [i for i in sp[ev] if sc[i] in trs]
    print(f"{ev}: n={len(sp[ev])} actives={int(lab[sp[ev]].sum())} exact canonical-SMILES duplicates in train: {len(dup)} (actives {int(lab[dup].sum()) if dup else 0}); Murcko-scaffold shared with train: {len(sdup)}")
print("valid-test scaffold overlap:", len(set(sc[sp['valid']]) & set(sc[sp['test']])))
print("train actives", int(lab[sp['train']].sum()), "rate", lab[sp['train']].mean().round(4), "valid rate", lab[sp['valid']].mean().round(4), "test rate", lab[sp['test']].mean().round(4))
