"""FoR47 Language, communication and culture — CoNLL-2018 UD dependency parsing, metric LAS (max).

Data: Universal Dependencies v2.2 (the CoNLL 2018 shared-task release, hdl 11234/1-2837, UD 2.2 licences), complete
CoNLL-U members extracted by the data team under ``<DATA_ROOT>/for47-ud-conll2018/subset/ud-treebanks-v2.2``.

* **Item** = one sentence of a treebank with **gold tokenization**: its syntactic words (FORM, in order) and
  multiword tokens (surface ranges such as French ``du`` = ``de le``); hidden label = HEAD and DEPREL of every
  syntactic word. Item id = ``ud/<treebank>/<official split>/<sent_id>``. Evaluation sentences have
  ``min_words``..``max_words`` words (default 3..40).
* **Pools.** IID treebank = ``UD_French-GSD``: src / val from the official *dev* file (hash partition 75 / 25 %
  by sentence), id from the official *test* file. OOD = ``UD_French-PUD`` (default; configurable): the CoNLL 2018
  *parallel* test treebank (news + Wikipedia sentences translated into French, annotated by a different team).
  As in CoNLL 2018, PUD has no training data, so OOD episodes see the source treebank's training data —
  ``lineage["ood_kind"] = "cross_treebank"`` (a different dataset of the same language).
* **Visible data (D_E).** Per episode a deterministic sample of ``n_train`` fully annotated sentences of the
  official *train* file of the IID treebank (as structured words and as CoNLL-U text) and a disjoint slice of
  ``n_dev`` train sentences whose trees are held by ``score_dev`` (the episode's ``_dev_evaluate`` is None).
  Train, dev and test files are disjoint by construction of UD. Tools ``read_conllu`` (CoNLL-U reader) and
  ``check_trees`` (the official evaluator's tree-validity rules) are provided.
* **Metric (D_V).** LAS of the official CoNLL 2018 evaluation script (``conll18_ud_eval.py`` v1.2): with gold
  tokenization every system word is aligned to its gold word, so LAS = #words whose HEAD and DEPREL match /
  #words, where DEPREL is compared on its universal part only (language-specific subtypes after ``:`` are
  ignored) and punctuation is included. Computed over all words of the episode (micro, as the official script
  does over a file); ``pooled_metric`` pools words over episodes. UAS is reported as auxiliary.
* **Reference baseline** (deterministic): right-branching chain (word i attaches to word i+1, the last word is
  the root) with, per word, the most frequent non-root DEPREL of its lower-cased form in the episode's training
  sample (global most frequent DEPREL for unseen forms). **Acceptance:** ``LAS >= reference LAS + margin``
  (default 0.15).
* **Hard constraints:** list of ``n_items`` parses; each parse has one integer head and one relation per word;
  every sentence is a valid tree as required by the official evaluator (heads in [0, n], exactly one root, no
  cycle); every relation's universal part is one of the 37 UD v2 relations.
"""
from __future__ import annotations

import collections
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib
from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for43_45_47_48_50 import (
    PARTITION_SEED, Lazy, PoolExhausted, as_list, c_list_length, check_split, draw_stratified, episode_id,
    episode_rng, ids_digest, norm_score, partition_groups, receipt_info, resolve_data_root, stable_int,
)

CODE = "FoR47"
FAMILY = "Humanities & law"
DATASET_DIR = "for47-ud-conll2018"
UD_DIR = Path("subset") / "ud-treebanks-v2.2"
UD_RELATIONS = ("acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj",
                "cop", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj",
                "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct",
                "reparandum", "root", "vocative", "xcomp")
DEV_FRACTIONS = {"src": 0.75, "val": 0.25}
_TB_FILE = {"UD_French-GSD": "fr_gsd", "UD_French-PUD": "fr_pud", "UD_Russian-Taiga": "ru_taiga",
            "UD_Coptic-Scriptorium": "cop_scriptorium"}


# ----------------------------------------------------------------------------------------------- CoNLL-U
@dataclass(frozen=True)
class Sentence:
    uid: str
    sent_id: str
    text: str
    forms: tuple[str, ...]
    lemmas: tuple[str, ...]
    upos: tuple[str, ...]
    xpos: tuple[str, ...]
    feats: tuple[str, ...]
    heads: tuple[int, ...]
    deprels: tuple[str, ...]
    mwts: tuple[tuple[int, int, str], ...]

    @property
    def n(self) -> int:
        return len(self.forms)


def parse_conllu(text: str, uid_prefix: str = "") -> list[Sentence]:
    """Read CoNLL-U text into sentences (multiword-token lines kept as ranges, empty nodes skipped)."""
    out: list[Sentence] = []
    cols: list[list[str]] = []
    mwts: list[tuple[int, int, str]] = []
    sid, stext = "", ""

    def flush() -> None:
        nonlocal cols, mwts, sid, stext
        if cols:
            k = len(out)
            sent_id = sid or str(k + 1)
            out.append(Sentence(f"{uid_prefix}{sent_id}", sent_id, stext,
                                tuple(c[1] for c in cols), tuple(c[2] for c in cols), tuple(c[3] for c in cols),
                                tuple(c[4] for c in cols), tuple(c[5] for c in cols),
                                tuple(int(c[6]) if c[6].lstrip("-").isdigit() else -1 for c in cols),
                                tuple(c[7] for c in cols), tuple(mwts)))
        cols, mwts, sid, stext = [], [], "", ""

    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line.strip():
            flush()
            continue
        if line.startswith("#"):
            m = re.match(r"#\s*sent_id\s*=\s*(.*)$", line)
            if m:
                sid = m.group(1).strip()
            m = re.match(r"#\s*text\s*=\s*(.*)$", line)
            if m:
                stext = m.group(1).strip()
            continue
        c = line.split("\t")
        if len(c) != 10:
            raise ValueError(f"CoNLL-U line does not have 10 tab-separated columns: {line[:80]!r}")
        if "." in c[0]:
            continue
        if "-" in c[0]:
            a, b = c[0].split("-")
            mwts.append((int(a), int(b), c[1]))
            continue
        cols.append(c)
    flush()
    return out


def to_conllu(sents: list[Sentence]) -> str:
    lines: list[str] = []
    for s in sents:
        lines.append(f"# sent_id = {s.sent_id}")
        lines.append(f"# text = {s.text}")
        starts = {a: (a, b, f) for a, b, f in s.mwts}
        for i in range(1, s.n + 1):
            if i in starts:
                a, b, f = starts[i]
                lines.append(f"{a}-{b}\t{f}\t_\t_\t_\t_\t_\t_\t_\t_")
            j = i - 1
            lines.append("\t".join([str(i), s.forms[j], s.lemmas[j], s.upos[j], s.xpos[j], s.feats[j], str(s.heads[j]),
                                    s.deprels[j], "_", "_"]))
        lines.append("")
    return "\n".join(lines) + "\n"


def sentence_dict(s: Sentence, with_tree: bool, index: int | None = None) -> dict:
    words = []
    for j in range(s.n):
        w: dict[str, Any] = {"id": j + 1, "form": s.forms[j]}
        if with_tree:
            w.update(lemma=s.lemmas[j], upos=s.upos[j], xpos=s.xpos[j], feats=s.feats[j], head=s.heads[j],
                     deprel=s.deprels[j])
        words.append(w)
    d: dict[str, Any] = {"text": s.text, "n_words": s.n, "words": words,
                         "multiword_tokens": [{"range": [a, b], "form": f} for a, b, f in s.mwts]}
    if with_tree:
        d = {"sent_id": s.sent_id, **d}
    if index is not None:
        d = {"index": index, **d}
    return d


# ----------------------------------------------------------------------------------------------- scoring
def coerce_parse(p: Any, n: int) -> tuple[list[int] | None, list[str] | None, str]:
    """A parse as (heads, deprels); accepts {"head": [...], "deprel": [...]} or [[head, deprel], ...]."""
    heads: Any
    rels: Any
    if isinstance(p, dict):
        heads, rels = p.get("head", p.get("heads")), p.get("deprel", p.get("deprels"))
    elif isinstance(p, (list, tuple)) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in p):
        heads, rels = [x[0] for x in p], [x[1] for x in p]
    else:
        return None, None, "parse must be {'head': [...], 'deprel': [...]}"
    if hasattr(heads, "tolist"):
        heads = heads.tolist()
    if hasattr(rels, "tolist"):
        rels = rels.tolist()
    if not isinstance(heads, (list, tuple)) or not isinstance(rels, (list, tuple)):
        return None, None, "head and deprel must be lists"
    if len(heads) != n or len(rels) != n:
        return None, None, f"expected {n} heads and deprels, got {len(heads)} and {len(rels)}"
    out_h: list[int] = []
    for h in heads:
        is_int = isinstance(h, (int, np.integer)) and not isinstance(h, (bool, np.bool_))
        is_whole_float = isinstance(h, (float, np.floating)) and float(h).is_integer()
        if not (is_int or is_whole_float):
            return None, None, f"head {h!r} is not an integer"
        out_h.append(int(h))
    if not all(isinstance(r, str) for r in rels):
        return None, None, "every deprel must be a string"
    return out_h, [str(r) for r in rels], ""


def tree_issues(heads: list[int], rels: list[str]) -> list[str]:
    """Official CoNLL 2018 validity rules: heads in [0, n], exactly one root, no cycle; + UD relation check."""
    n = len(heads)
    issues: list[str] = []
    if any(h < 0 or h > n for h in heads):
        issues.append("HEAD outside the sentence")
        return issues
    if sum(1 for h in heads if h == 0) != 1:
        issues.append(f"{sum(1 for h in heads if h == 0)} roots (exactly one word must have HEAD 0)")
    for i in range(1, n + 1):
        seen = set()
        j = i
        while j != 0:
            if j in seen:
                issues.append("cycle")
                break
            seen.add(j)
            j = heads[j - 1]
        if issues and issues[-1] == "cycle":
            break
    bad = sorted({r for r in rels if r.split(":")[0] not in UD_RELATIONS})
    if bad:
        issues.append(f"relations not in UD v2: {bad[:5]}")
    return issues


def attachment_counts(gold: Sentence, heads: list[int], rels: list[str]) -> tuple[int, int]:
    """(#LAS-correct, #UAS-correct) words; DEPREL compared on the universal part (official LAS)."""
    las = uas = 0
    for gh, gr, h, r in zip(gold.heads, gold.deprels, heads, rels):
        if gh == h:
            uas += 1
            if gr.split(":")[0] == r.split(":")[0]:
                las += 1
    return las, uas


def chain_baseline(sent: Sentence, rel_of_form: dict[str, str], fallback: str) -> tuple[list[int], list[str]]:
    n = sent.n
    heads = [i + 1 if i < n else 0 for i in range(1, n + 1)]
    rels = ["root" if h == 0 else rel_of_form.get(f.lower(), fallback) for h, f in zip(heads, sent.forms)]
    return heads, rels


def form_relation_table(train: list[Sentence]) -> tuple[dict[str, str], str]:
    by_form: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    allc: collections.Counter = collections.Counter()
    for s in train:
        for f, r in zip(s.forms, s.deprels):
            u = r.split(":")[0]
            if u != "root":
                by_form[f.lower()][u] += 1
                allc[u] += 1
    table = {f: sorted(c.items(), key=lambda x: (-x[1], x[0]))[0][0] for f, c in by_form.items()}
    fallback = sorted(allc.items(), key=lambda x: (-x[1], x[0]))[0][0] if allc else "dep"
    return table, fallback


# ----------------------------------------------------------------------------------------------- data
@dataclass
class _Data:
    sents: dict[str, Sentence]
    eval_split: dict[str, dict[str, list[str]]]     # split -> stratum ("all") -> uids
    train_ids: list[str]


def _read(path: Path, prefix: str) -> list[Sentence]:
    return parse_conllu(path.read_text(encoding="utf-8"), prefix)


def _tb_path(root: Path, tb: str, part: str) -> Path:
    return root / DATASET_DIR / UD_DIR / tb / f"{_TB_FILE.get(tb, tb)}-ud-{part}.conllu"


def _build(root: Path, iid_tb: str, ood_tb: str, partition_seed: int, min_words: int, max_words: int,
           max_train_words: int) -> _Data:
    sents: dict[str, Sentence] = {}

    def load(tb: str, part: str) -> list[Sentence]:
        ss = _read(_tb_path(root, tb, part), f"ud/{tb}/{part}/")
        for s in ss:
            if s.uid in sents:
                raise ValueError(f"duplicate sentence id {s.uid}")
            sents[s.uid] = s
        return ss

    def ok(s: Sentence) -> bool:
        return min_words <= s.n <= max_words and not tree_issues(list(s.heads), list(s.deprels))

    train = [s for s in load(iid_tb, "train") if 2 <= s.n <= max_train_words]
    dev = [s for s in load(iid_tb, "dev") if ok(s)]
    test = [s for s in load(iid_tb, "test") if ok(s)]
    ood = [s for s in load(ood_tb, "test") if ok(s)]
    part = partition_groups({s.uid: s.uid for s in dev}, DEV_FRACTIONS, f"{CODE}|{partition_seed}|{iid_tb}|dev")
    ev = {"src": {"all": sorted(part["src"])}, "val": {"all": sorted(part["val"])},
          "id": {"all": sorted(s.uid for s in test)}, "ood": {"all": sorted(s.uid for s in ood)}}
    eval_forms = {s.forms for sp in ev.values() for u in sp["all"] for s in [sents[u]]}
    # guard against duplicated sentences across files: visible training never contains an evaluation sentence
    train_ids = sorted(s.uid for s in train if s.forms not in eval_forms)
    return _Data(sents, ev, train_ids)


# ----------------------------------------------------------------------------------------------- adapter
class UDParsingAdapter:
    """TaskAdapter for FoR47 (see module docstring)."""

    discipline = CODE
    name = "CoNLL-2018-UD"
    family = FAMILY
    metric = "LAS"
    direction = "max"
    task_type = "dependency_parsing"

    def __init__(self, data_root: str | None = None, partition_seed: int = PARTITION_SEED,
                 iid_treebank: str = "UD_French-GSD", ood_treebank: str = "UD_French-PUD", n_train: int = 1000,
                 n_dev: int = 64, min_words: int = 3, max_words: int = 40, margin: float = 0.15,
                 budget: Budget | None = None, **_: Any) -> None:
        self.root = resolve_data_root(data_root)
        self.partition_seed = int(partition_seed)
        self.iid_tb, self.ood_tb = str(iid_treebank), str(ood_treebank)
        self.n_train, self.n_dev = int(n_train), int(n_dev)
        self.min_words, self.max_words = int(min_words), int(max_words)
        self.margin = float(margin)
        self.budget = budget
        self._data: Lazy[_Data] = Lazy(lambda: _build(self.root, self.iid_tb, self.ood_tb, self.partition_seed,
                                                      self.min_words, self.max_words, 80))

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        need = [_tb_path(self.root, self.iid_tb, p) for p in ("train", "dev", "test")] + \
               [_tb_path(self.root, self.ood_tb, "test")]
        missing = [str(p) for p in need if not p.is_file()]
        if missing:
            return False, f"missing UD 2.2 CoNLL-U files: {missing}"
        return True, f"UD 2.2 {self.iid_tb} (IID) and {self.ood_tb} (OOD) under {self.root / DATASET_DIR / UD_DIR}"

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 1:
            raise ValueError("items_per_episode must be >= 1")
        data = self._data.get()
        rng = episode_rng(CODE, split, seed, self.partition_seed)
        draws = draw_stratified(data.eval_split[split], int(n), int(items_per_episode), rng, f"{CODE}/{split}")
        return [self._episode(data, split, k, int(seed), items) for k, items in enumerate(draws)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """LAS over all words of all episodes (invalid outputs scored with the reference parses)."""
        correct = total = 0
        for p in per_episode:
            if not p:
                continue
            w = list(p.get("n_words") or [])
            c = p.get("las_correct")
            if c is None:
                c = p.get("ref_las_correct")
            if c is None or len(c) != len(w):
                raise ValueError("FoR47 pooled payload needs n_words and las_correct (or ref_las_correct)")
            correct += int(sum(c))
            total += int(sum(w))
        return correct / total if total else None

    # ------------------------------------------------------------ episode
    def _visible(self, data: _Data, ep_key: str) -> tuple[list[str], list[str]]:
        """Seeded (train sample, dev slice) of the training file; dev sentences obey the evaluation word limits."""
        rng = np.random.default_rng(stable_int(CODE, "visible", ep_key))
        ids = data.train_ids
        perm = [ids[j] for j in rng.permutation(len(ids))]
        dev: list[str] = []
        train: list[str] = []
        for u in perm:
            if len(dev) < self.n_dev and self.min_words <= data.sents[u].n <= self.max_words:
                dev.append(u)
            elif len(train) < self.n_train:
                train.append(u)
            if len(dev) >= self.n_dev and len(train) >= self.n_train:
                break
        return sorted(train), sorted(dev)

    def _episode(self, data: _Data, split: str, k: int, seed: int, items: list[str]) -> Episode:
        eid = episode_id(CODE, split, seed, k)
        pool = "ood" if split == "ood" else "iid"
        ev = [data.sents[u] for u in items]
        n_items = len(ev)
        tr_ids, dev_ids = self._visible(data, eid)
        tr = [data.sents[u] for u in tr_ids]
        dv = [data.sents[u] for u in dev_ids]
        n_tr, n_dev = len(tr), len(dv)
        table, fallback = form_relation_table(tr)
        n_words = [s.n for s in ev]

        train_rows = [sentence_dict(s, True) for s in tr]
        train_text = to_conllu(tr)
        dev_rows = [sentence_dict(s, False, j) for j, s in enumerate(dv)]
        eval_rows = [sentence_dict(s, False, j) for j, s in enumerate(ev)]

        def score_parses(gold: list[Sentence], parses: list) -> tuple[float, float, list[int]]:
            las = uas = tot = 0
            per: list[int] = []
            for g, p in zip(gold, parses):
                h, r, why = coerce_parse(p, g.n)
                if h is None or r is None:
                    raise ValueError(f"invalid parse for sentence {len(per)}: {why}")
                a, b = attachment_counts(g, h, r)
                las, uas, tot = las + a, uas + b, tot + g.n
                per.append(a)
            return las / max(tot, 1), uas / max(tot, 1), per

        ref_parses = [chain_baseline(s, table, fallback) for s in ev]
        ref_counts = [attachment_counts(s, h, r) for s, (h, r) in zip(ev, ref_parses)]
        ref_las = sum(c[0] for c in ref_counts) / max(sum(n_words), 1)
        dev_ref = Lazy(lambda: score_parses(dv, [{"head": h, "deprel": r} for h, r in
                                                 (chain_baseline(s, table, fallback) for s in dv)])[0])

        # ---- D_E tools
        def load_train(inputs: dict, config: dict) -> dict:
            return {"train": copy.deepcopy(train_rows), "train_conllu": train_text}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_sentences": copy.deepcopy(dev_rows)}

        def score_dev(inputs: dict, config: dict) -> dict:
            parses, why = as_list(inputs.get("dev_parses"))
            if parses is None or len(parses) != n_dev:
                raise ValueError(f"dev_parses must be a list of {n_dev} parses ({why})")
            las, uas, _ = score_parses(dv, parses)
            return {"dev_las": las, "dev_uas": uas, "dev_reference_las": float(dev_ref.get())}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"sentences": copy.deepcopy(eval_rows)}

        def pretrained_parse(inputs: dict, config: dict) -> dict:
            """Run the frozen French parser as an explicit task tool (no fitting or hidden labels)."""
            sentences = inputs.get("sentences")
            if not isinstance(sentences, list):
                raise ValueError("sentences must be a list of gold-tokenized sentence dictionaries")
            from scilib import udparse_pretrained
            if not udparse_pretrained.available():
                raise RuntimeError("the offline Stanza/CamemBERT parser is unavailable on this worker")
            parses = udparse_pretrained.parse_gold_tokens(sentences)
            return {"parses": [{"head": list(p["head"]), "deprel": list(p["deprel"])} for p in parses]}

        def read_conllu(inputs: dict, config: dict) -> dict:
            text = inputs.get("conllu")
            if not isinstance(text, str):
                raise ValueError("conllu must be a string in CoNLL-U format")
            return {"sentences": [sentence_dict(s, True) for s in parse_conllu(text)]}

        def check_trees(inputs: dict, config: dict) -> dict:
            parses, why = as_list(inputs.get("parses"))
            if parses is None:
                raise ValueError(f"parses must be a list ({why})")
            lens = inputs.get("n_words")
            lens = list(lens) if lens is not None else [None] * len(parses)
            valid, issues = [], []
            for p, nw in zip(parses, lens):
                n = int(nw) if nw is not None else (len(p.get("head", [])) if isinstance(p, dict) else len(p))
                h, r, msg = coerce_parse(p, n)
                iss = [msg] if h is None or r is None else tree_issues(h, r)
                valid.append(not iss)
                issues.append("; ".join(iss))
            return {"valid": np.asarray(valid, dtype=bool), "issues": issues}

        sent_doc = "dict: index, text, n_words, words [{id, form}], multiword_tokens [{range, form}]"
        tools = [
            ToolSpec("load_train", f"{n_tr} fully annotated sentences of the official {self.iid_tb} training file "
                     "(UD 2.2): per word id, form, lemma, upos, xpos, feats, head, deprel; also the same sentences "
                     "as CoNLL-U text.",
                     {}, {"train": PortSchema("list", (n_tr,), dtype="dict", description="annotated sentences"),
                          "train_conllu": PortSchema("text", description="CoNLL-U")},
                     load_train),
            ToolSpec("load_dev_inputs", f"{n_dev} further training-file sentences (gold tokenization only; trees "
                     "withheld); score parses of them with score_dev.",
                     {}, {"dev_sentences": PortSchema("list", (n_dev,), dtype="dict", description=sent_doc)},
                     load_dev_inputs),
            ToolSpec("score_dev", "Official LAS and UAS (fractions) of dev_parses (one parse per dev sentence, same "
                     "order, format as y) against the withheld dev trees, plus the LAS of the adapter's reference "
                     "parser.",
                     {"dev_parses": PortSchema("list", (n_dev,), dtype="dict")},
                     {"dev_las": PortSchema("number", unit="1"), "dev_uas": PortSchema("number", unit="1"),
                      "dev_reference_las": PortSchema("number", unit="1")},
                     score_dev),
            ToolSpec("load_eval_inputs", f"The {n_items} evaluation sentences with gold tokenization (trees hidden), "
                     "in the order that y must follow.",
                     {}, {"sentences": PortSchema("list", (n_items,), dtype="dict", description=sent_doc)},
                     load_eval_inputs),
            ToolSpec("pretrained_parse", "Frozen French dependency parser (Stanza French-GSD with a CamemBERT-large "
                     "encoder). Parses gold-tokenized sentences without reading their hidden trees or fitting on the "
                     "episode; its training-data overlap and cross-treebank limitations are documented in the task.",
                     {"sentences": PortSchema("list", ("m",), dtype="dict", description=sent_doc)},
                     {"parses": PortSchema("list", ("m",), dtype="dict", description="predicted HEAD/DEPREL")},
                     pretrained_parse),
            ToolSpec("read_conllu", "CoNLL-U reader: parses CoNLL-U text into sentences (same structure as "
                     "load_train entries; multiword-token lines become ranges, empty nodes are skipped).",
                     {"conllu": PortSchema("text")}, {"sentences": PortSchema("list", ("m",), dtype="dict")},
                     read_conllu),
            ToolSpec("check_trees", "Validity of parses under the official CoNLL 2018 evaluator rules (heads in "
                     "[0, n], exactly one root, no cycle) and the UD v2 relation inventory; n_words optional.",
                     {"parses": PortSchema("list", ("m",), dtype="dict"),
                      "n_words": PortSchema("list", ("m",), dtype="int")},
                     {"valid": PortSchema("array", ("m",), dtype="bool"), "issues": PortSchema("list", ("m",), dtype="str")},
                     check_trees),
        ]

        # ---- D_V constraints
        def _parsed(yv: Any) -> tuple[list[tuple[list[int], list[str]]] | None, str]:
            lst, why = as_list(yv)
            if lst is None:
                return None, why
            if len(lst) != n_items:
                return None, f"len(y)={len(lst)} != {n_items}"
            out = []
            for j, (p, s) in enumerate(zip(lst, ev)):
                h, r, msg = coerce_parse(p, s.n)
                if h is None or r is None:
                    return None, f"sentence {j}: {msg}"
                out.append((h, r))
            return out, ""

        def c_format(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            ps, why = _parsed(yv)
            return ps is not None, (why or "every parse has one integer head and one relation per word")

        def c_tree(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            ps, why = _parsed(yv)
            if ps is None:
                return False, why
            bad = {j: iss for j, (h, r) in enumerate(ps) if (iss := [i for i in tree_issues(h, r)
                                                                     if not i.startswith("relations")])}
            return not bad, ("all sentences are valid trees" if not bad else
                             f"invalid trees: {dict(list(bad.items())[:5])}")

        def c_rel(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            ps, why = _parsed(yv)
            if ps is None:
                return False, why
            bad = sorted({r for _, rs in ps for r in rs if r.split(":")[0] not in UD_RELATIONS})
            return not bad, ("all relations are UD v2 relations" if not bad else f"unknown relations {bad[:8]}")

        constraints = [
            c_list_length(n_items, "one parse per evaluation sentence"),
            ConstraintSpec("parse_format", "every entry of y is {'head': [int]*n_words, 'deprel': [str]*n_words} for "
                           "its sentence", c_format),
            ConstraintSpec("valid_tree", "every sentence's heads form a tree: heads in [0, n_words], exactly one word "
                           "with head 0, no cycle (official CoNLL 2018 evaluator rules)", c_tree),
            ConstraintSpec("ud_relations", "the universal part (before ':') of every deprel is one of the 37 UD v2 "
                           "relations", c_rel),
        ]

        # ---- D_V evaluator
        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            payload = {"item_ids": list(items), "n_words": list(n_words), "las_correct": None,
                       "uas_correct": None, "ref_las_correct": [c[0] for c in ref_counts]}
            base = {"reference": ref_las, "reference_name": "right-branching chain + per-form majority relation",
                    "margin": self.margin, "n_items": n_items, "n_words": int(sum(n_words))}
            ps, why = _parsed(yv)
            if ps is None:
                return EvalResult(metrics={"reference_las": ref_las}, primary=None, direction="max", accepted=False,
                                  details={**base, "norm_score": 0.0, "pooled_payload": payload, "invalid": why})
            counts = [attachment_counts(s, h, r) for s, (h, r) in zip(ev, ps)]
            tot = max(sum(n_words), 1)
            las = sum(c[0] for c in counts) / tot
            uas = sum(c[1] for c in counts) / tot
            payload["las_correct"] = [c[0] for c in counts]
            payload["uas_correct"] = [c[1] for c in counts]
            return EvalResult(metrics={"las": las, "uas": uas, "reference_las": ref_las}, primary=las, direction="max",
                              accepted=bool(las >= ref_las + self.margin),
                              details={**base, "norm_score": norm_score(las, ref_las, "max"), "pooled_payload": payload})

        tb = self.ood_tb if pool == "ood" else self.iid_tb
        lang = tb.split("_", 1)[-1].split("-")[0]
        objective = (
            "Language, communication and culture - dependency parsing in the Universal Dependencies (UD v2) "
            f"framework (CoNLL 2018 shared task data). Each evaluation item is one {lang} sentence from the "
            f"treebank {tb}, given with gold tokenization: its syntactic words in order (multiword tokens such as "
            "French 'du' are already split into their syntactic words, and their surface ranges are listed). The "
            "hidden target is the gold basic dependency tree: for every word its HEAD (index of the governing word, "
            "1-based; 0 for the root) and its DEPREL (UD relation label).\n"
            f"Visible data: load_train returns {n_tr} fully annotated sentences from the {self.iid_tb} training "
            f"file; load_dev_inputs returns {n_dev} further training sentences without trees and score_dev scores "
            "parses of them; load_eval_inputs returns the evaluation sentences; pretrained_parse is an explicit "
            "frozen-parser tool for those visible sentences; its predictions are returned under the `parses` output "
            "port and must be submitted directly as y. For a formal tool-on episode, add pretrained_parse, wire "
            "its `parses` port to submit.y, and finish immediately; do not add a code/fine-tuning node, score_dev "
            "probe, or later rewire after that submit, because replacing the tool output invalidates the formal "
            "lineage gate. read_conllu reads CoNLL-U text; check_trees applies the "
            "official tree-validity rules.\n"
            f"Deliverable y: a list of {n_items} parses; y[i] = {{'head': [...], 'deprel': [...]}} for "
            "sentences[i] from load_eval_inputs (same order), with one integer head and one relation per word, in "
            "word order.\n"
            "Evaluation: labeled attachment score (LAS) of the official CoNLL 2018 evaluator over all words of the "
            "episode: a word counts as correct when both its head and its relation match the gold tree; relation "
            "subtypes after ':' are ignored; punctuation is included. Every sentence must be a valid tree (heads in "
            "[0, n], exactly one root, no cycles).\n"
            + scilib.describe("udparse") + scilib.describe_extra("udparse_pretrained")
        )
        lineage = {
            "dataset": "Universal Dependencies (CoNLL 2018 shared task release)", "version": "UD 2.2",
            "source_url": "https://hdl.handle.net/11234/1-2837",
            "license": "per treebank (UD 2.2 licences; fr_gsd CC BY-SA 4.0, fr_pud CC BY-SA 3.0)",
            "receipt": receipt_info(self.root / DATASET_DIR / "receipt.json"),
            "pool": pool, "treebank": tb,
            "pool_source": (f"{self.ood_tb} official test file" if pool == "ood" else
                            f"{self.iid_tb} official {'test' if split == 'id' else 'dev'} file"),
            "ood_kind": "cross_treebank" if pool == "ood" else None,
            "ood_shift": (f"different treebank ({self.ood_tb}: parallel news/Wikipedia sentences, different "
                          "annotators) evaluated with the source treebank's training data, as in CoNLL 2018")
            if pool == "ood" else None,
            "split": split, "split_seed": seed, "partition_seed": self.partition_seed, "index": k,
            "item_ids": list(items), "n_items": n_items, "n_words": int(sum(n_words)),
            "word_limits": [self.min_words, self.max_words],
            "train_source": f"{self.iid_tb} official train file", "n_train": n_tr, "n_dev": n_dev,
            "train_ids_sha256": ids_digest(tr_ids), "dev_item_ids": list(dev_ids),
            "rebuilt_split": True, "historical_ids_recovered": False,
        }
        return Episode(
            id=eid, discipline=CODE, family=FAMILY, split=split, task_type=self.task_type, objective=objective,
            required_output=PortSchema("list", (n_items,), dtype="dict",
                                       description="{'head': [int], 'deprel': [str]} per evaluation sentence"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=2400.0, max_node_s=600.0,
                                         max_llm_items=8 * max(n_items, n_dev)),
            lineage=lineage,
            acceptance=(f"LAS >= reference LAS + {self.margin:g}; reference = right-branching chain with per-form "
                        "majority relation from the visible training sentences"),
            tolerance={"rtol": 1e-6, "atol": 1e-8},
            tags=[CODE, "linguistics", "syntax", "dependency-parsing", "Universal-Dependencies", "CoNLL-U", "LAS",
                  lang.lower(), tb, pool],
            metric=self.metric, direction=self.direction, n_items=n_items,
            _evaluate=evaluate, _dev_evaluate=None,
        )


Adapter = UDParsingAdapter

__all__ = ["Adapter", "UDParsingAdapter", "parse_conllu", "to_conllu", "tree_issues", "attachment_counts",
           "coerce_parse", "chain_baseline", "form_relation_table", "UD_RELATIONS", "PoolExhausted"]
