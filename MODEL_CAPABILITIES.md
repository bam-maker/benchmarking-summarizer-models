# Model capabilities and operating-range evidence

This document synthesizes existing evidence; no inference was repeated for this
pass. **A useful factual-quality range is unestablished for all four models.**
We can specify reproducible operating windows and observed behaviors, but cannot
certify summaries even for short articles. Keep the existing demo behavior until
independent review supports a change. No quality-default model is recommended.

## Evidence and its status

- **Measured:** the inspected CNN/DailyMail 3.0.0 validation rows 2-41 are now
  **development data**, not an untouched final test. All 160 application Medium
  outputs completed, with all source characters presented and natural EOS.
  Thirty-eight articles fit one call; rows 33/39 used two source chunks per model.
  No additional generative consolidation was needed in this sample.
- **Measured:** the fixed five-case audit made 172 controlled direct calls. With
  identical input/settings, all 24 raw chunk outputs and all 20 merged summaries
  reproduced the saved app outputs. Direct inference is already established;
  these calls were not repeated. Checkpoint-native versus adaptive controls vary
  one length/EOS setting at a time. Stanford versus approximate preprocessing and
  400/1024 prefixes are separate interventions.
- **AI-assisted observations:** existing source/reference reading identifies
  omitted main events, wrong entities/relationships, unsupported authorship,
  altered attribution, repetition and incomplete clauses. These are provisional
  evidence-linked judgments, not human scores or population error rates.
- **Awaiting human review:** correctness/severity judgments, acceptable operating
  ranges, preference usefulness and any quality-default recommendation. The new
  anonymous packet has blank human-rating fields.

Evidence: [original development results](reports/validation_40_results.json),
[40-source review](reports/validation_40_review.html),
[controlled audit](reports/factual_audit_review.html), and
[before/after diagnosis](reports/quality_regression.json). Original reports remain
unchanged. Rows 0/1 and the unverified Jensen text were earlier development cases;
none is an independent validation of a new setting.

## Systems, conventions and reproducible baselines (experiment A)

All four systems are English CNN/DailyMail news summarizers. These are
**complete-system comparisons**: size, pretraining, data preparation and decoding
differ, preventing architecture-only claims.

| Model and purpose | Checkpoint and size | Training input convention | Architectural/implementation capacity |
|---|---|---|---|
| LSTM: smaller recurrent attention/copy baseline | Official OpenNMT `ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20`; 42,789,205 parameters; 1-layer bidirectional encoder, 1-layer decoder | Tokenized lowercase non-anonymized CNN/DM; source first 400 tokens, target 100 | No fixed positional table in recurrent encoder/decoder. Current app work window 1024, output cap 256; this is a workload policy, not a learned quality boundary |
| Transformer: conventional Transformer/copy comparison | Official OpenNMT `sum_transformer_model_acc_57.25_ppl_9.22_e16`; 81,468,757 parameters; 4 encoder/4 decoder layers | Same 400/100 training convention | Saved encoder/decoder positional buffers each length 5000; adapter conservatively allows at most 4999 decoder steps. App uses 1024 input/256 output; 5000-token quality was not tested |
| BART: broadly pretrained denoising encoder-decoder | `facebook/bart-large-cnn`, revision `37f520fa929c961707657b28798b30c003dd100b`; 406,290,432 parameters; 12/12 layers | Cased untokenized article followed by original byte-level BPE, truncated source within 1024-position configuration | Configured encoder/decoder 1024 positions; source counts include two special tokens. Adapter conservative output limit 1023 new tokens; app cap 512 |
| PEGASUS: summarization-oriented pretraining comparison | `google/pegasus-cnn_dailymail`, revision `40d588fdab0cc077b80d950b300bf66ad3c75b92`; 570,797,056 instantiated parameters, 568,699,904 learned; 16/16 layers | Original case-preserving SentencePiece; released CNN/DM task parameters specify input 1024, target/decode 128 | Configured encoder/decoder 1024 positions, source includes EOS. Adapter conservative output limit 1023; app cap 512. Fixed sinusoidal tables are not newly trained parameters |

Sizes/positional buffers come from existing local verification traces, not a new
model load. Provenance and weight hashes: [OpenNMT manifest](OpenNMT_Verified_Setup/checkpoint_manifest.json)
and [HF manifest](model_manifest.json). BART's documented recipe preserves case,
uses byte-level BPE and truncates source; batch `max-tokens=2048` is not an article
context limit. The released recipe/configuration does not independently prove
every detail of the exact checkpoint's training history.
[BART recipe](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.summarization.md).
PEGASUS task input/output conventions are documented in the
[original task parameters](https://github.com/google-research/pegasus/blob/main/pegasus/params/public_params.py).

**Preprocessing:** OpenNMT's historical data preparation uses Stanford CoreNLP
3.7 PTB tokenization, then lowercase, original-line period repair, original
checkpoint vocabularies and dynamic source dictionaries for copying. Target
sentences use `<t>`/`</t>`; do not insert them into article text. The app's NLTK
Treebank approximation differs at quoted sentence endings and abbreviations.
The audit used the original tokenizer on identical flattened HF text, but could
not reconstruct missing original `.story` line boundaries/caption repairs.
[Original data preparation](https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py).
BART/PEGASUS use their own revision-pinned tokenizers and preserve case; no legacy
lowercasing, shared replacement tokenizer or swapped vocabulary is appropriate.

| Baseline | Deterministic native decoding | Difference from current application |
|---|---|---|
| OpenNMT LSTM/Transformer | Beam 5, min 35/max 100 decoder tokens, Wu alpha .9, summary coverage beta 5, stepwise penalty, trigram blocking except `.`, `<t>`, `</t>`; no unknown-token replacement | Same search profile; app caps minimum by source tokens/4 and derives maximum from preference with native headroom. App source window 1024; fixed-budget API still 400 |
| BART | Beam 4, length penalty 2, min_length 56/max_length 142 including decoder start, no-repeat 3, early stopping true; checkpoint BOS/EOS/PAD retained | App min_new_tokens is bounded native minimum minus 1; maximum derives from preference but at least 141, at most 512; forced EOS disabled |
| PEGASUS | Beam 8, length penalty .8, min_length 32/max_length 128 including start, no-repeat 0, early stopping false; own special tokens | App bounded min_new_tokens and maximum at least 127, at most 512; forced EOS disabled; native search otherwise inherited |

OpenNMT has no HF-style embedded generation config: baseline means the pinned
0.2.1 explicit CNN/DM command plus translator default max100. Its prose says
alpha .8, command .9; use documented command .9. Current website beam10 is not
the pinned recipe. See [pinned documentation](OpenNMT_Verified_Setup/legacy/docs/source/Summarization.md).

Preserve **literal checkpoint defaults** as reproducibility controls. HF defaults
can manufacture EOS at a maximum. The natural-EOS baseline changes only
`forced_eos_token_id=None`; audit controls already measured this distinction.
At a cap without natural EOS, record an unfinished diagnostic, not a completed
sentence. Do not extend a baseline silently. Checkpoint floors may be excessive
for tiny inputs; they are study settings rather than product recommendations.
The complete configs/control graph are saved in
[baseline_protocol.json](reports/model_capabilities/baseline_protocol.json).

Direct A inference requires no chunking when source fits its selected window.
An OpenNMT article above400 can have a clearly labeled historical-prefix A control
and a separate expanded full-input control; they have different coverage. HF
articles above1024 cannot be a complete single pass in this implementation.
Keep prefix diagnostics distinct from experiment C.

## Evaluated ranges: post-hoc description, not validated quality thresholds

Groups use **each model's input tokenizer**, including special tokens where
applicable: short <=400, medium 401-800, long single-pass 801-1024, beyond window
>1024. These group names do not mean output Short/Medium/Detailed. Same underlying
40 article IDs, but group membership differs by tokenizer. All output settings
were application Medium. No cross-length causal claim is supported by comparing
different stories. No sample falls in several gaps between observed endpoints.

| Model | Short n; observed tokens | Medium n; tokens | Long single-pass n; tokens | Beyond window n; whole-source tokens |
|---|---|---|---|---|
| LSTM / Transformer | 20; 99-387 | 14; 406-629 | 4; 858-917 | 2; 1400-1590 |
| BART | 19; 114-393 | 15; 413-688 | 4; 916-989 | 2; 1486-1720 |
| PEGASUS | 19; 115-378 | 15; 403-642 | 4; 887-967 | 2; 1493-1719 |

Source word ranges by group: legacy81-341/348-541/733-831/1188-1354;
HF81-303/341-541/733-831/1188-1354. Whole-source counts beyond1024 are NOT the
tokens in one call; each of the two chunks fits its own model window.

| Model | Short RL / mean words / generation s | Medium RL / words / s | Long single-pass RL / words / s | C: beyond-window RL / words / s |
|---|---|---|---|---|
| LSTM | 30.50 / 34.5 / .399 | 25.37 / 38.7 / .522 | 10.38 / 53.8 / .906 | 18.78 / 111.5 / 1.598 |
| Transformer | 36.69 / 32.8 / .575 | 29.39 / 34.9 / .649 | 15.97 / 40.5 / .926 | 21.89 / 74.0 / 1.313 |
| BART | 34.68 / 46.0 / .683 | 28.00 / 49.3 / .818 | 17.37 / 57.5 / .772 | 23.57 / 97.5 / 1.526 |
| PEGASUS | 34.72 / 37.7 / .920 | 29.39 / 34.1 / .907 | 12.52 / 44.5 / .947 | 27.80 / 77.5 / 2.130 |

RL is flat ROUGE-L macro F1 x100 with Porter stemming, not Lsum or factuality.
[CSV](reports/model_capabilities/input_ranges.csv) includes R1/R2/L, repetition,
word ranges and punctuation flags; [JSON](reports/model_capabilities/input_ranges.json)
includes exact IDs and existing AI notes per group. Larger-input groups have only
4/2 articles. Lower overlap there does not establish an intrinsic length effect.
The two template-reference pages remain in primary development metrics; the
unchanged [labeled sensitivity](reports/validation_40_results.json) excludes only
those two templates for description, not poorly performing outputs.

All160 succeed technically; zero exact duplicate sentences or capped unfinished
returns. Mean excess-trigram percentages over40: LSTM0, Transformer0, BART.09,
PEGASUS.63; PEGASUS row30 repeats a claim. Missing-final-punctuation flags are
3/0/2/2, but LSTM row34 ends in an incomplete clause even with a period and EOS.
This distinction matters for readability and completeness assessment.

Existing latency is same RTX4090, float32, synchronized generation, one run per
case without explicit warmup. Cold load/transfer separately: .486/1.146/1.805/3.141s
in model order. **Peak/allocated/reserved per-case VRAM was not measured** in these
reports. Do not turn parameter counts into claimed memory measurements. Lazy
loading/one-active-GPU-model behavior is established execution evidence. Earlier
synthetic fixed-budget speed results use a different protocol and are not pooled.

### Two different OpenNMT window questions

1. **Same short content:** row21 is99 app tokens/98 Stanford tokens.400 and1024
   admit identical complete tokens, use the same35/100 decoding, and produce
   identical raw IDs for each model/tokenizer pair. Raising a numeric ceiling
   alone changes neither input nor quality in this example. It still misses the
   bombing event. No broad beneficial-window claim follows from this one case.
2. **Additional context above400:** the same full stories supply400 versus1024
   prefixes, with decoding/preprocessing held fixed. Row19 Transformer NLTK400
   removes the reversal but leaves a wrong scorer; Stanford400 reverses the
   result again. LSTM gets expanded input right but both400 versions wrong.
   Evidence supports context sensitivity, not a blanket rollback or quality gain.
   Prefix token coverage/omissions are recorded in the audit. Row39 exceeds both
   prefixes: neither prefix control is a full-article summary.

## Strengths and failure patterns: AI-assisted source observations

| Model | Observed useful behavior | Failures and evidence |
|---|---|---|
| LSTM | Smallest system; lowest mean generation latency in this development run; expanded row19 correctly states Real's defeat | Misattributes Holder's quote to Obama (35), researcher quote to Pope (41), loses bombing event (21), malformed ending (34); Monohm retained but Runcible omitted in39 |
| Transformer | Conventional pretrained CNN/DM comparison; highest overlap in this sample; preserves some core responsibility/injury facts | Reversed win/loss and wrong scorer/league (19), invented opinion author (35), short bombing omission (21), both main product names omitted (39). ROUGE lead is not a quality-default justification |
| BART | Coherent Gomis collapse/recovery summary (2); correct match roles (19); names product/company in two-chunk39 | Invents Frida Ghitis (35) under native and adaptive decoding; omits bombing (21); some background replaces current event details. More words than peers do not certify usefulness |
| PEGASUS | Correct match roles (19), clear recovery update (37), named Runcible/Monohm in39; summarization-oriented pretraining comparison | Invents Eric Liu (35), repeats Cambodia claim (30) even with native settings, omits bombing (21), some weakened certainty/attribution (41) |

The main-event, entity, unsupported-claim and attribution assessments require
human confirmation. Do not infer factuality by counting entity overlap or by
converting these selected observations into fabricated error rates. References
are not complete factual oracles:35's byline is absent from source; templates
10/14 include generic reference wording absent from their source pages.

## Length as preference (experiment B)

Current recoverable app ratios15/25/35% generate approximate budgets, not required
output lengths. Current code accepts EOS, never pads to those percentages and
does not cut sentences at a word count. Keep Short/Medium/Detailed in the UI;
there is **no independently validated model-specific preset to promote yet**.

Existing controls already narrow the search: on rows19/30/35 and both chunks39,
native/default and adaptive length settings produce identical raw IDs in every
model. On short21, relaxing the native minimum changes length and secondary
detail but fails to recover the main bombing event. BART grows25 to41 words at
native minimum without restoring it; PEGASUS20 to23. Longer ceilings did not
force additional detail. These are useful negative findings, not a reason to
force three different texts. Earlier synthetic presets are historical behavior,
not an independent quality validation of the current policy.

For a future single predeclared candidate per model: use natural-EOS A as Medium;
test Short by lowering only its minimum, and Detailed by increasing only its
maximum within decoder/work limits. Keep native search, avoid filler and preserve
unchanged outputs as valid outcomes. Numeric candidates must be frozen before
new inference. Judge supported facts/omissions/readability against A; a longer
output or higher ROUGE cannot pass the gate. No candidate is recommended or run
here, because present evidence supports retaining behavior while collecting human
judgments. Reserved rows42-47 remain unused; if a candidate is justified later,
save those separate IDs, compare once, and do not tune repeatedly on that check.

## Long-article application method (experiment C)

Keep full-article processing and all coverage/workload safeguards. It is an
application extension beyond the selected single-pass range, not the checkpoint
baseline or a proven quality improvement. Audit source chunks before merging.
In39, Transformer loses product names in its first raw chunk; exact-order merge
does not remove them. BART/PEGASUS retain names but keep tangential second-chunk
material. Both original chunks are processed; preserving text coverage does not
preserve all facts. Extra generative consolidation is absent in this example.

Rows33/39 give only two genuine multi-chunk articles; older repeated-fixture and
termination checks do not validate news factuality at greater depth. Changes to
chunk boundaries, budgets or consolidation need a separate C study with source
facts traced to each intermediate and final output. Do not rank A, B and C
together. No additional mechanism is introduced here.

## Human review packet and next decisions

Open [anonymous review](reports/model_capabilities/blind_review.html) and copy
[blank ratings](reports/model_capabilities/human_ratings_BLANK.csv) to a
reviewer-specific filename. Six purposively selected cases provide24 summaries:
one plausible success and the five diagnosed cases, with randomized per-case
model order. This is a small qualitative comparison, not a prevalence sample.
The packet omits identities, scores and AI notes. Keep
`reports/model_capabilities/blinding_key_DO_NOT_SHARE_WITH_REVIEWER.json` separate
until ratings are complete. Prior exposure and recognizable casing/styles limit
blinding; outputs were not rewritten to hide those cues.

Instructions define main-event correctness0-2, entity/relationship and unsupported
claim counts, important omissions0-2, readability1-5, attribution/uncertainty,
repetition/incomplete text and evidence citations. Read source before reference;
do not repair source uncertainty using outside knowledge. Leave all human ratings
blank until a person enters them. Ideally use two independent reviewers and
discuss disagreements after initial ratings; the project currently has none.

Operating recommendations are conservative **study boundaries**, not factuality
guarantees: OpenNMT400 for historical direct-baseline work; expanded401-1024 is
an application diagnostic range requiring human assessment. HF<=1024 supports
direct full-input operation; factual usefulness within that window remains
unestablished. Above these ranges use explicitly partial prefix diagnostics or
separate C full-article processing. No model wins by ROUGE alone.

## Preservation and reproduction

[Application snapshot](reports/model_capabilities/application_snapshot.zip) stores
the nine frozen application/configuration/manifest/requirements files byte for
byte, verified against [snapshot manifest](reports/model_capabilities/snapshot_manifest.json).
Weights and environment remain at their existing paths. The ZIP is intentionally
Git-ignored by the repository's archive policy: preserve it locally with the
project backup. Extract to a separate directory to review before restoring;
do not overwrite subsequent work blindly. Existing [recovery instructions](PROJECT_STATUS.md)
cover checkpoints/environment. Baseline configs are separately preserved in JSON.

New outputs live under `reports/model_capabilities/`; original results are intact.
The standard-library-only synthesis script performs no inference or downloads:

```powershell
.\.venv\Scripts\python.exe tests\synthesize_capabilities.py
```

It verifies protected hashes, snapshot bytes, six cases/24 blind records and
blank ratings. Copy the blank CSV before filling it; regenerating this packet
recreates the blank template. No production code, UI, dependencies, checkpoints,
benchmark method or final test was changed; no push or slides were created.
