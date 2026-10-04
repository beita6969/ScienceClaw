# FoR36 SOTA36B SCNet formal attempt — 2026-10-04

`separate_scnet` is a promising frozen route: on the trusted visible-dev
capture it reached **8.528417 dB**, versus **8.091974 dB** for `htdemucs_ft`.
The FoR36 ID pool has nine episodes; the original SOTA36 consumed 00–03, so
this batch reserved the disjoint 04–07 suffix.

The batch did not produce a model result. The first launch used the existing
single-card service on port 22200, whose marker declared model `sc-llm`, while
the launcher defaulted to `sc-llm-l3`. All four episodes stopped at policy
step 0 with HTTP 404 (`The model sc-llm-l3 does not exist`); no tool call,
scorer call, or prediction was made. The validator consequently found
`completed=false`, no `separate_scnet` record, and no hard-valid result.

These four items are recorded as an infrastructure/configuration failure and
are not rerun under the one-evaluation-per-item rule. The launcher is fixed in
commits `64bb53e` and `73f30ac`: it discovers the endpoint's `/v1/models`
identifier when `MODEL` is unset and refuses to rerun any formal output that
already contains a result receipt. The fix is synchronized to Leonardo and
passes shell syntax checks. The SCNet visible-dev result remains engineering
evidence only; no new formal score is claimed.

## SOTA36C final reserved item (ID-08)

After the launcher fix, the last unused ID item was run once through the live
single-card endpoint (`59264793`, auto-discovered model `sc-llm`). This attempt
was not an HTTP/configuration failure: the agent completed 24 steps but never
called `separate_scnet` or produced an output, then stopped at `step_budget`.
The receipt is therefore `completed=false`, `primary=null`, `z=0`, with no
scorer evidence. ID-08 is consumed under the one-evaluation-per-item rule and
is not rerun. FoR36 consequently has no new SCNet formal score; the 8.528417 dB
value remains visible-dev engineering evidence only.
