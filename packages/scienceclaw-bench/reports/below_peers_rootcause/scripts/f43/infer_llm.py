"""Greedy OCRonos (Llama-3-8B, PleIAs) correction of OCR pieces. usage: infer_llm.py <model_dir> <units_ocr.json> <out.jsonl> <chunk_chars> [batch]
units_ocr.json = {uid: {"lang": .., "ocr": ..}}; pieces are cut at whitespace; output rows {uid, k, ocr, pred}."""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, inp, outp, chunk = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
bs = int(sys.argv[5]) if len(sys.argv) > 5 else 16


def pieces(text, n):
    pos, out = 0, []
    while pos < len(text):
        end = min(len(text), pos + n)
        if end < len(text):
            k = max(text.rfind(" ", pos + n // 2, end + 40), text.rfind("\n", pos + n // 2, end + 40))
            end = k + 1 if k > pos else end
        out.append(text[pos:end]); pos = end
    return out


units = json.load(open(inp))
rows = [{"uid": u, "k": k, "lang": v["lang"], "ocr": p} for u, v in units.items() for k, p in enumerate(pieces(v["ocr"], chunk))]
tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
net = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda().eval()
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ocr"]))
res = [None] * len(rows); t0 = time.time()
for s in range(0, len(order), bs):
    idx = order[s:s + bs]
    prompts = ["### Text ###\n" + rows[i]["ocr"].rstrip() + "\n\n### Correction ###\n" for i in idx]
    x = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        out = net.generate(**x, max_new_tokens=int(x.input_ids.shape[1] * 1.25) + 32, do_sample=False, stop_strings=["#END#"], tokenizer=tok,
                           pad_token_id=tok.pad_token_id)
    for i, o in zip(idx, tok.batch_decode(out[:, x.input_ids.shape[1]:], skip_special_tokens=True)):
        res[i] = o.split("#END#")[0].strip()
    if (s // bs) % 10 == 0:
        print(s, len(order), f"{time.time() - t0:.0f}s", flush=True)
with open(outp, "w") as f:
    for r, o in zip(rows, res):
        f.write(json.dumps({**r, "pred": o}, ensure_ascii=False) + "\n")
print("done", f"{time.time() - t0:.0f}s", flush=True)
