"""Write the visible FoR39 training data (load_train output) and the dev replica inputs to an npz for server-side experiments."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scienceclaw.bench.tasks.for39_eedi import Adapter  # noqa: E402

out = Path(sys.argv[1])
ad = Adapter()
ep = ad.build_episodes("val", 1, 1)[0]
tools = {t.name: t for t in ep.tools}
tr = tools["load_train"].fn({}, {})
dv = tools["load_dev_inputs"].fn({}, {})
subj = tr["question_subjects"]
S = 1 + max(max(s) for s in subj if s)
qs = np.zeros((len(subj), S), dtype=np.int8)
for q, s in enumerate(subj):
    qs[q, list(s)] = 1
np.savez_compressed(out, answers=tr["answers"].astype(np.int8), values=tr["answer_values"].astype(np.int8),
                    meta=tr["student_meta"].to_numpy(dtype=np.float32), qsubj=qs,
                    dev_can=dv["dev_can_query"], dev_tgt=dv["dev_targets"], dev_meta=dv["dev_student_meta"].to_numpy(dtype=np.float32))
print({k: v.shape for k, v in np.load(out).items()})
