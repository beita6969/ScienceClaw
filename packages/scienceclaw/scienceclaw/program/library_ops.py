"""Typed Operators for the main entry points of the tool library.

Each operator is a single ``code`` node that calls one ``scilib`` function, with port schemas (type, shape, unit) and pre/post
conditions that the executor checks on the delivered values. They make the pretrained models first-class building blocks of a
workflow canvas: the policy wires an operator instead of writing import and calling code, and a contract violation is reported
as a diagnostic. They start every program as library components (``provenance.source == "library"``).
"""
from __future__ import annotations

from typing import Any

from scienceclaw.core.graph import Node, WorkflowGraph
from scienceclaw.core.operators import Contract, OperatorSpec
from scienceclaw.core.schema import PortSchema

NODE = "call"


def _p(type_: str, desc: str, shape: tuple | None = None, unit: str | None = None, dtype: str | None = None) -> PortSchema:
    return PortSchema(type=type_, shape=shape, unit=unit, dtype=dtype, description=desc)


# id, name, tool, description, inputs, outputs, body of run(), pre, post, applicability tags
_SPECS: list[dict[str, Any]] = [
    dict(id="chronos_quantile_forecast", tool="tsfm.forecast",
         description="Quantile forecasts (0.1 / 0.5 / 0.9) of univariate series from the pretrained Chronos-2 model; the series only serve as context.",
         inputs={"histories": _p("list", "list of 1-D float sequences, oldest value first (NaN = missing, at least 3 observed values)"),
                 "horizon": _p("number", "number of future steps to forecast (1..1024)", unit="1")},
         outputs={"quantiles": _p("array", "predicted quantiles, in the units of the input", shape=("n", "horizon", 3), dtype="float")},
         code="from scilib import tsfm\nout = tsfm.forecast(inputs['histories'], int(inputs['horizon']), quantiles=(0.1, 0.5, 0.9), model='chronos_2')\nreturn {'quantiles': out}",
         pre=[{"port": "histories", "check": "nonempty"}], post=[{"port": "quantiles", "check": "finite"}],
         tags=["forecasting", "time series", "pretrained", "chronos"]),
    dict(id="audio_embedding_ast", tool="audioenc.embed",
         description="Clip embeddings of audio waveforms from the pretrained Audio Spectrogram Transformer (AudioSet), pooled output.",
         inputs={"waveforms": _p("array", "float waveforms in [-1, 1], one clip per row (zero-padded)", shape=("n_clips", "n_samples"), dtype="float"),
                 "sample_rate": _p("number", "sampling rate of the waveforms", unit="Hz")},
         outputs={"embeddings": _p("array", "one vector per clip (not normalised)", shape=("n_clips", "d"), dtype="float")},
         code="from scilib import audioenc\nout = audioenc.embed(inputs['waveforms'], sample_rate=float(inputs['sample_rate']), model='ast_audioset', kind='pooled')\nreturn {'embeddings': out}",
         pre=[{"port": "waveforms", "check": "finite"}], post=[{"port": "embeddings", "check": "finite"}],
         tags=["audio", "embedding", "pretrained", "ast"]),
    dict(id="text_embedding_bge", tool="textenc.embed",
         description="L2-normalised sentence embeddings from the pretrained BGE-large English encoder (cosine = dot product).",
         inputs={"texts": _p("list", "list of strings (at most 512 tokens each)")},
         outputs={"embeddings": _p("array", "unit-norm sentence embeddings", shape=("n_texts", 1024), dtype="float")},
         code="from scilib import textenc\nreturn {'embeddings': textenc.embed(inputs['texts'], model='bge_large_en')}",
         pre=[{"port": "texts", "check": "nonempty"}], post=[{"port": "embeddings", "check": "finite"}],
         tags=["text", "embedding", "retrieval", "pretrained", "bge"]),
    dict(id="text_relevance_bge_reranker", tool="textenc.relevance",
         description="Cross-encoder relevance logits of every (query, passage) pair from the pretrained BGE reranker.",
         inputs={"passages": _p("list", "list of passages"), "queries": _p("list", "list of queries")},
         outputs={"scores": _p("array", "relevance logit, higher = more relevant", shape=("n_passages", "n_queries"), dtype="float")},
         code="from scilib import textenc\nreturn {'scores': textenc.relevance(inputs['passages'], inputs['queries'], model='bge_reranker_large')}",
         pre=[{"port": "passages", "check": "nonempty"}, {"port": "queries", "check": "nonempty"}], post=[{"port": "scores", "check": "finite"}],
         tags=["text", "retrieval", "ranking", "pretrained", "bge"]),
    dict(id="text_entailment_deberta", tool="textenc.nli",
         description="Natural-language-inference log-probabilities over (contradiction, entailment, neutral) for every (premise, hypothesis) pair.",
         inputs={"premises": _p("list", "list of premises"), "hypotheses": _p("list", "list of hypotheses")},
         outputs={"log_probs": _p("array", "log-probabilities; last axis = (contradiction, entailment, neutral)", shape=("n_premises", "n_hypotheses", 3), dtype="float")},
         code="from scilib import textenc\nreturn {'log_probs': textenc.nli(inputs['premises'], inputs['hypotheses'], model='nli_deberta_v3_large')}",
         pre=[{"port": "premises", "check": "nonempty"}, {"port": "hypotheses", "check": "nonempty"}], post=[{"port": "log_probs", "check": "finite"}],
         tags=["text", "inference", "entailment", "verification", "pretrained", "deberta"]),
    dict(id="protein_site_logprobs_esm2", tool="proteinplm.site_logprobs",
         description="Natural-log probabilities of the 20 amino acids at every position of a protein sequence from the pretrained ESM-2 650M model (masked marginals).",
         inputs={"wild_type": _p("text", "one amino-acid sequence (single-letter codes)")},
         outputs={"logprobs": _p("array", "log-probability of each of the 20 residues (columns in scilib.proteinplm.AA order)", shape=("length", 20), dtype="float")},
         code="from scilib import proteinplm\nreturn {'logprobs': proteinplm.site_logprobs(inputs['wild_type'], masked=True, model='esm2_650m')}",
         pre=[{"port": "wild_type", "check": "nonempty"}], post=[{"port": "logprobs", "check": "finite"}],
         tags=["protein", "variant effect", "language model", "pretrained", "esm2"]),
    dict(id="music_source_separation_htdemucs", tool="audiosep_pretrained.separate_pretrained",
         description="Separate stereo music excerpts into vocals, drums, bass and other with the pretrained Hybrid Transformer Demucs.",
         inputs={"mixtures": _p("array", "stereo mixtures", shape=("k", "n", 2), dtype="float"),
                 "sample_rate": _p("number", "sampling rate of the mixtures", unit="Hz")},
         outputs={"estimates": _p("array", "estimates; axis 1 = (vocals, drums, bass, other)", shape=("k", 4, "n", 2), dtype="float")},
         code="from scilib import audiosep_pretrained\nout = audiosep_pretrained.separate_pretrained(inputs['mixtures'], model='htdemucs', sample_rate=int(inputs['sample_rate']))\nreturn {'estimates': out}",
         pre=[{"port": "mixtures", "check": "finite"}], post=[{"port": "estimates", "check": "finite"}],
         tags=["audio", "music", "source separation", "pretrained", "demucs"]),
    dict(id="french_dependency_parse_stanza", tool="udparse_pretrained.parse_gold_tokens",
         description="Universal-Dependencies tags and dependency parses for already tokenised French sentences with the pretrained CamemBERT/Stanza pipeline.",
         inputs={"sentences": _p("list", "list of sentences, each a list of word forms")},
         outputs={"parses": _p("list", "one dict per sentence with UPOS, lemma, head and relation per word")},
         code="from scilib import udparse_pretrained\nreturn {'parses': udparse_pretrained.parse_gold_tokens(inputs['sentences'])}",
         pre=[{"port": "sentences", "check": "nonempty"}], post=[{"port": "parses", "check": "nonempty"}],
         tags=["linguistics", "parsing", "french", "universal dependencies", "pretrained", "stanza"]),
    dict(id="phonon_features_mlip", tool="matphonon_mlip.phonon_features",
         description="Harmonic phonon-spectrum features of crystal structures from a pretrained universal interatomic potential (SevenNet).",
         inputs={"structures": _p("list", "list of crystal structures (lattice, species, coordinates)")},
         outputs={"features": _p("dict", "{'X': float array (n, n_features), 'names': [str]}; frequencies in cm^-1")},
         code="from scilib import matphonon_mlip\nreturn {'features': matphonon_mlip.phonon_features(inputs['structures'], model='sevennet')}",
         pre=[{"port": "structures", "check": "nonempty"}], post=[],
         tags=["materials", "phonons", "interatomic potential", "pretrained", "sevennet"]),
    dict(id="image_embedding_clip", tool="clip_retrieval.encode_images",
         description="L2-normalised image embeddings from the frozen CLIP ViT-B/32 checkpoint.",
         inputs={"images": _p("array", "uint8 RGB images", shape=("n", "H", "W", 3), dtype="int")},
         outputs={"embeddings": _p("array", "unit-norm image embeddings", shape=("n", 512), dtype="float")},
         code="from scilib import clip_retrieval\nreturn {'embeddings': clip_retrieval.encode_images(inputs['images'], model='open_clip_vit_b32')}",
         pre=[{"port": "images", "check": "nonempty"}], post=[{"port": "embeddings", "check": "finite"}],
         tags=["vision", "image", "embedding", "retrieval", "pretrained", "clip"]),
    dict(id="field_panoptic_segmentation_m2f", tool="phenoseg_m2f.predict_panoptic",
         description="Hierarchical panoptic segmentation of top-down field images (soil / crop / weed, plant and leaf instances) with the pretrained Mask2Former models.",
         inputs={"images": _p("array", "uint8 RGB field images", shape=("n", "H", "W", 3), dtype="int")},
         outputs={"panoptic": _p("dict", "arrays (n, H, W): semantics (0 soil, 1 crop, 2 weed), plant_instances, leaf_instances")},
         code="from scilib import phenoseg_m2f\nreturn {'panoptic': phenoseg_m2f.predict_panoptic(inputs['images'])}",
         pre=[{"port": "images", "check": "nonempty"}], post=[],
         tags=["agriculture", "segmentation", "images", "pretrained", "mask2former"]),
]


def _build(spec: dict[str, Any]) -> OperatorSpec:
    body = WorkflowGraph()
    body.nodes[NODE] = Node(id=NODE, kind="code", code="def run(inputs, config):\n    " + spec["code"].replace("\n", "\n    ") + "\n",
                            inputs=dict(spec["inputs"]), outputs=dict(spec["outputs"]))
    return OperatorSpec(
        id=spec["id"], version=1, name=spec["id"], description=spec["description"], body=body,
        inputs=dict(spec["inputs"]), outputs=dict(spec["outputs"]),
        input_map={k: [(NODE, k)] for k in spec["inputs"]}, output_map={k: (NODE, k) for k in spec["outputs"]},
        contract=Contract(pre=list(spec["pre"]), post=list(spec["post"]),
                          applicability={"tools": [spec["tool"]], "input_types": sorted({p.type for p in spec["inputs"].values()})}),
        provenance={"source": "library", "tool": spec["tool"]}, tags=list(spec["tags"]))


def library_operators() -> dict[str, OperatorSpec]:
    """The typed operators wrapping the pretrained tools, keyed by operator id."""
    return {s["id"]: _build(s) for s in _SPECS}
