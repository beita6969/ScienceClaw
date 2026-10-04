"""CheMeleon fingerprints (chemprop >= 2.2, weights chemeleon_mp.pt from Zenodo 10.5281/zenodo.15460715) for every ogbg-molhiv SMILES.
usage: embed.py <smiles.txt> <chemeleon_mp.pt> <out.npy>   (follows chemeleon_fingerprint.py of the CheMeleon repository)"""
import sys, time
import numpy as np, torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

smi = [l.rstrip("\n") for l in open(sys.argv[1])]
ck = torch.load(sys.argv[2], weights_only=True)
mp = nn.BondMessagePassing(**ck["hyper_parameters"]); mp.load_state_dict(ck["state_dict"])
model = MPNN(message_passing=mp, agg=nn.MeanAggregation(), predictor=RegressionFFN(input_dim=mp.output_dim)).eval().cuda()
fz = featurizers.SimpleMoleculeMolGraphFeaturizer()


def mol(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        m = Chem.MolFromSmiles(s, sanitize=False)
        if m is not None:
            try:
                m.UpdatePropertyCache(strict=False); Chem.FastFindRings(m)
            except Exception:
                return None
    return m


out = np.zeros((len(smi), mp.output_dim), dtype=np.float32); bad = []; t0 = time.time()
for i in range(0, len(smi), 512):
    graphs, idx = [], []
    for j in range(i, min(i + 512, len(smi))):
        try:
            m = mol(smi[j]); graphs.append(fz(m)); idx.append(j)
        except Exception:
            bad.append(j)
    bmg = BatchMolGraph(graphs); bmg.to(device=model.device)
    with torch.no_grad():
        out[idx] = model.fingerprint(bmg).numpy(force=True)
    if (i // 512) % 10 == 0:
        print(i, len(smi), f"{time.time() - t0:.0f}s", flush=True)
np.save(sys.argv[3], out)
print("done", out.shape, "unparsed", bad, f"{time.time() - t0:.0f}s", flush=True)
