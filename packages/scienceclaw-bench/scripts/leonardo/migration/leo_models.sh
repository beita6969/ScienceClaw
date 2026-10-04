#!/bin/bash
# Stage tool weights on Leonardo straight from their sources (HF / fbaipublicfiles); layout = $L/models/<subdir> as scilib._pretrained expects.
set -u
F=/leonardo_scratch/fast/AIFAC_F02_774/rqian000; L=/leonardo_scratch/large/userexternal/rqian000
M=$L/models; mkdir -p $M $L/sc-tools/logs
HF=$F/envs/sc-vllm/bin/hf; export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME=$L/cache/hf
dl() {  # repo dest [revision]
  echo "== $1 -> $2"; nice -n 10 $HF download "$1" --local-dir "$M/$2" ${3:+--revision "$3"} --exclude "*.msgpack" --exclude "*.h5" --exclude "*.ot" --exclude "flax_*" --exclude "tf_*" --exclude "onnx/*" --exclude "*.onnx" --exclude "openvino/*" --exclude "*.bin" 2>&1 | tail -n 2
}
rm -rf $M/tsfm $M/audioenc $M/textenc $M/esm2_650m $M/sam2hf $M/dinov2-large $M/dinov2-base $M/camembert-large
dl amazon/chronos-2 tsfm/chronos_2
dl amazon/chronos-bolt-base tsfm/chronos_bolt_base
dl MIT/ast-finetuned-audioset-10-10-0.4593 audioenc/ast_audioset
dl laion/clap-htsat-unfused audioenc/clap_htsat_unfused
dl BAAI/bge-large-en-v1.5 textenc/bge_large_en
dl BAAI/bge-reranker-large textenc/bge_reranker_large
dl MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli textenc/nli_deberta_v3_large
dl facebook/esm2_t33_650M_UR50D esm2_650m
dl facebook/sam2.1-hiera-large sam2hf
dl facebook/dinov2-large dinov2-large
dl facebook/dinov2-base dinov2-base
dl almanach/camembert-large camembert-large
for s in depparse/gsd_camembert-large.pt pos/gsd_camembert-large.pt lemma/gsd_nocharlm.pt pretrain/conll17.pt forward_charlm/newswiki.pt backward_charlm/newswiki.pt; do
  mkdir -p $M/stanza_fr/fr/$(dirname $s); nice -n 10 curl -fL --retry 5 -C - -o $M/stanza_fr/fr/$s https://huggingface.co/stanfordnlp/stanza-fr/resolve/v1.14.0/models/$s
done
mkdir -p $M/phenobench_weyler; nice -n 10 curl -fL --retry 5 -C - -o $M/phenobench_weyler/weyler_checkpoint_0381.pth https://www.ipb.uni-bonn.de/html/projects/phenobench/hierarchical/weyler/weyler_checkpoint_0381.pth
mkdir -p $M/demucs; nice -n 10 curl -fL --retry 5 -C - -o $M/demucs/955717e8-8726e21a.th https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th; printf "models: ['955717e8']\n" > $M/demucs/htdemucs.yaml
for f in f7e0c4bc-ba3fe64a.th d12395a8-e57c48e6.th 92cfc3b6-ef3bcb9c.th 04573f0d-f3cf25b2.th; do
  nice -n 10 curl -fL --retry 5 -C - -o $M/demucs/$f https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/$f
done
printf "models: ['f7e0c4bc', 'd12395a8', '92cfc3b6', '04573f0d']\nweights: [\n  [1., 0., 0., 0.],\n  [0., 1., 0., 0.],\n  [0., 0., 1., 0.],\n  [0., 0., 0., 1.],\n]\n" > $M/demucs/htdemucs_ft.yaml
# The official MDX-extra ensembles are complementary to HTDemucs on MUSDB18
# and use the same frozen Demucs loader.  Stage all four members plus their
# immutable bag manifests; no task data or labels are involved.
for f in e51eebcc-c1b80bdd.th a1d90b5c-ae9d2452.th 5d2d6c55-db83574e.th cfa93e08-61801ae1.th; do
  nice -n 10 curl -fL --retry 5 -C - -o $M/demucs/$f https://dl.fbaipublicfiles.com/demucs/mdx_final/$f
done
printf "models: ['e51eebcc', 'a1d90b5c', '5d2d6c55', 'cfa93e08']\nsegment: 44\n" > $M/demucs/mdx_extra.yaml
mkdir -p $M/sam2; nice -n 10 curl -fL --retry 5 -C - -o $M/sam2/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
echo MODELS_DONE; du -sh $M/* | sort -rh | head -30
