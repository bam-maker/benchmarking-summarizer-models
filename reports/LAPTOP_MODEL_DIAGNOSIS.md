# Laptop model diagnosis: latest Riverton demo

**Scope:** The fictional article supplied by the user on 2026-11-22, using the existing Ubuntu CPU environment, checkpoints and `natural-checkpoint-v1` demo policy. This is **not** the different article in [`tests/fixtures/riverton.txt`](../tests/fixtures/riverton.txt) or the earlier [`RIVERTON_DIAGNOSIS.md`](RIVERTON_DIAGNOSIS.md). This diagnosis changed no production setting, checkpoint, dependency or benchmark.

## Finding

The two **reported display strings are exactly identical**. They match the Transformer result on this article, but **not** the LSTM result from either direct inference or a fresh request to the running Gradio `/summarize` route. The actual LSTM and Transformer decoded token sequences differ at “approximately” and throughout the last sentence. One model's raw text was therefore not made identical by cleanup, chunking or merging. The original browser interaction was not recorded with its response text or selected value, so the reason its LSTM display showed Transformer text remains **unresolved**. A stale browser result, a mismatched selection at submit time, or transcription are possibilities, not demonstrated causes. Current backend routing and model caching work for this article.

The running app's console log recorded **315 words, 2,202 characters, one chunk and one model call** for the latest requests. The supplied body is exactly 315 words/2,202 characters (SHA-256 `648c2a202c8912f6d1fc5bbe468162d2670be52508916bbbf4ac7fd7e5457fa8`); including its title would be 321 words/2,249 characters. Thus the title was almost certainly absent from the original textbox. That inference uses counts because the log did not retain the source text. Reproduction below uses the body only. The title was not sent as an additional model call because the body counts and three exact output matches resolve the input question for this case.

| Model | Supplied UI output | Direct backend raw result | One-chunk pipeline display | Check |
|---|---|---|---|---|
| LSTM | Transformer text: “approximately 18,000 … the town will cover the expense” | **Different:** “18,000 … two schools closed Wednesday … reopened Thursday” | Same as direct; a fresh live Gradio `/summarize` request also returned this distinct LSTM text | Original UI mismatch **not reproduced** |
| Transformer | “approximately 18,000 … the town will cover the expense” | Same words, with OpenNMT `<t>` sentence tags | Exact supplied UI string | Reproduced |
| BART | Valve failure, pressure drop, precaution, two negative tests | Same text between special tokens | Exact supplied UI string | Reproduced |
| PEGASUS | “Riverton officials lifted …” | Same words with `<n>` line separators | Exact supplied UI string after line separators become spaces | Reproduced |

Each direct call bypassed Gradio and invoked its selected backend with the app's current checkpoint profile. An independent pipeline call for each model produced the same raw IDs as its direct call. All four processed the full 2,202 characters in **one chunk**, made **one model call**, and stopped at a generated EOS, before the output cap. There was no truncation, retry, consolidation or sentence deletion in this example. The app's `deduplicate_sentences` only joins the single PEGASUS output's line separators here. The exact article, reported outputs and reproduced outputs appear below; local raw IDs and token traces from this investigation are in ignored `.cache/latest_riverton_*.json` while this workspace is retained.

**Routing and identity:** [`app.py`](../app.py) connects the article textbox and model dropdown, in that order, to `generate_summary`; that handler passes the selected name to [`article_summarizer.py`](../article_summarizer.py), which calls `get_model(name)`. [`summarizer.py`](../summarizer.py) caches by the exact model name under a lock. A minimal same-process check found keys `LSTM` and `Transformer`, reuse of the LSTM object on a second lookup, and distinct backend/model objects for the two names. They resolve to separate local `.pt` files and distinct `RNNEncoder/InputFeedRNNDecoder` versus `TransformerEncoder/TransformerDecoder` objects, both with `CopyGenerator`. BART and PEGASUS resolve to their own pinned Hugging Face snapshots and different model classes. The button clears the previous result, dropdown/article edits invalidate it, and controls are disabled during generation. These checks and the fresh live LSTM route show no reproducible cache-key or dispatch swap. They cannot reconstruct the original browser state or exclude a one-time display race without a captured browser response.

| Model | Resolved checkpoint / class | Input tokens | Actual decoding for this article | Output tokens / stop |
|---|---|---:|---|---|
| LSTM | `OpenNMT_Verified_Setup/lstm_inference.pt` / `onmt.models.model.NMTModel` with `RNNEncoder` | 359 | beam 5; min 35/max 100; Wu length α=.9, coverage β=5, trigram block | 54 / EOS |
| Transformer | `OpenNMT_Verified_Setup/transformer_inference.pt` / `onmt.models.model.NMTModel` with `TransformerEncoder` | 359 | same OpenNMT search settings | 45 / EOS |
| BART | `facebook/bart-large-cnn` at `37f520fa929c961707657b28798b30c003dd100b` / `BartForConditionalGeneration` | 405 | beam 4; length penalty 2; no-repeat trigram; min 49/max 141 **new** tokens; no sampling | 59 / EOS |
| PEGASUS | `google/pegasus-cnn_dailymail` at `40d588fdab0cc077b80d950b300bf66ad3c75b92` / `PegasusForConditionalGeneration` | 368 | beam 8; length penalty .8; no no-repeat n-gram rule; min 31/max 127 **new** tokens; no sampling | 46 / EOS |

OpenNMT inference-file SHA-256 values in this Linux workspace are `2fb2bbb3ae529083572eac758736f30b54796335cc5a221d3e11746f5d6a2842` (LSTM) and `de23b61a2d8ec094447d04c475978afd31a61703c160bb2c6054e28b57d115a5` (Transformer). These are derived Linux-readable files; the original downloads' official hashes and tensor/vocabulary equivalence are documented in [`checkpoint_manifest.json`](../OpenNMT_Verified_Setup/checkpoint_manifest.json) and [`final_model_verification.json`](final_model_verification.json). Hugging Face IDs, revisions and weight-file identities are in [`model_manifest.json`](../model_manifest.json); the prior verification report records snapshot hashes. This investigation loaded those existing files offline and made no replacements.

## What each summary says about this article

The source's main event is **the Friday lifting of the boil-water advisory**, made possible by **completed valve repair, flushing, and two clear consecutive testing rounds**. The original warning was precautionary; contamination was not confirmed.

| Output | Main event and evidence | Accuracy/omissions |
|---|---|---|
| LSTM, reproduced | Mentions affected residents, pressure risk and the schools' closure/reopening. | Omits lifting, completed repairs and tests. No unsupported claim or entity confusion identified. The originally reported LSTM string is the Transformer string, so judge that display as such. |
| Transformer | Mentions affected residents, pressure risk and emergency-reserve funding. | Omits lifting, repairs and tests. Details are supported, but a reader cannot learn the current advisory status. |
| BART | Correctly says the valve failed, pressure dropped, the warning was precautionary and two consecutive test rounds found no harmful bacteria. | **Raw generation itself omits that the advisory was lifted**, and omits completed repairs/flushing. It does not explicitly say the advisory remains active; its omission leaves that status ambiguous. No invented fact, entity swap or repetition found. EOS occurred at 59 tokens, so neither a cap nor UI cleanup removed the main event. Why beam search chose those source facts rather than the lifted-advisory sentence is not identifiable from this one output. |
| PEGASUS | Clearly says officials **lifted** the advisory Friday and names the affected residents and valve failure. | Best captures this example's main event, but omits completed repairs and both negative test rounds; the reason lifting was safe is missing. No invented fact, entity swap or repetition found. This one preference is not a general ranking. |

## How these actual checkpoints work

All four tokenize the article, encode its words or subword pieces, and generate a summary one output token at a time while consulting the encoded article. The two OpenNMT checkpoints use lowercase word tokens and an **explicit learned copy gate**: at each step it mixes a vocabulary distribution with an attention-weighted distribution over a dynamic dictionary of words in this article. BART and PEGASUS use their own case-preserving subword tokenizers and standard encoder–decoder attention; they can reproduce source wording via their learned vocabulary predictions, but **do not have this OpenNMT copy gate**. Attention tells a decoder what source positions to use; it does not guarantee choosing the most important fact.

| Checkpoint | Verified architecture and training | Useful traits and limits on this laptop |
|---|---|---|
| **LSTM** · 42,789,205 parameters | One-layer **bidirectional LSTM** encoder reads the sequence in both directions; one-layer input-feeding LSTM decoder carries a recurrent state from word to word. 512 hidden units, 128-dimensional word embeddings, MLP/global attention. Its copy gate reuses that attention. Released OpenNMT CNN/DailyMail **summarization-trained** checkpoint; no broad pretraining is documented. | Smallest weights (~171 MB FP32 parameter floor), generally easiest to keep in 15 GiB RAM. Sequential recurrence limits parallelism and can lose important long-distance facts; it omitted the lifting here. |
| **Transformer** · 81,468,757 parameters | Four encoder and four decoder layers, 512-dimensional states, eight attention heads, 2,048-dimensional feed-forward blocks and position encodings. Encoder self-attention relates source positions; decoder masked self-attention plus cross-attention generates the next token. Its `CopyGenerator` uses the decoder's final cross-attention map in this legacy implementation, even though a separate copy-attention module is constructed. Released OpenNMT CNN/DailyMail **summarization-trained** checkpoint; no broad pretraining is documented. | ~326 MB FP32 parameter floor; more computation than LSTM here, and still missed the main event. Parallel attention helps relate distant words but costs memory as input grows. |
| **BART** · 406,290,432 parameters | Twelve-layer encoder and twelve-layer decoder, 1,024-dimensional states and 16 heads. Broad **denoising pretraining** teaches it to reconstruct corrupted text; this released `bart-large-cnn` checkpoint was then fine-tuned on CNN/DailyMail summarization. Uses byte-level BPE and learned positional embeddings. | Fluent, flexible wording; ~1.63 GB FP32 parameter floor plus activations, cache and Python overhead. On this article its raw selection of true background facts omitted the resolution. Slow CPU load/generation relative to the smaller models. |
| **PEGASUS** · 570,797,056 instantiated parameters, 568,699,904 learned | Sixteen-layer encoder and sixteen-layer decoder, 1,024-dimensional states and 16 heads. **Gap-sentence pretraining** removes important sentences and trains the model to generate them; this `cnn_dailymail` checkpoint was then fine-tuned for that task. Uses SentencePiece and fixed sinusoidal positional tables (the apparent missing position weights at load are those deterministic tables, not missing learned weights). | Captured the lifting here; still omitted repairs/tests. ~2.28 GB FP32 instantiated-parameter floor plus overhead, the heaviest and slowest practical CPU choice of these four. |

These are **whole-system** differences, including tokenizer, training, size and decoding, so the example cannot isolate architecture as the cause. Architecture and provenance are checked against local checkpoint metadata/adapter code and [OpenNMT's original summarization documentation](https://opennmt.net/OpenNMT-py/examples/summary/Summarization.html), the [BART paper](https://aclanthology.org/2020.acl-main.703/) and [checkpoint card](https://huggingface.co/facebook/bart-large-cnn), and the [PEGASUS paper](https://proceedings.mlr.press/v119/zhang20ae.html) and [checkpoint card](https://huggingface.co/google/pegasus-cnn_dailymail). The paper descriptions explain pretraining; the local checkpoints and pinned IDs establish what this app actually loads.

**Input limits:** The historical OpenNMT training recipe used first-400 source tokens and target length 100; 400 is a training convention, not an LSTM architectural maximum. The current app allows a 1,024-token work window for each OpenNMT backend; the Transformer's saved positional tables are length 5,000, which is not a 5,000-token quality claim. BART and PEGASUS configs allow 1,024 input positions including their special tokens. For longer articles the app makes sentence-aware chunks, passes every source character to a model, joins results in source order, removes duplicate sentences and may consolidate further. Complete **input coverage does not guarantee fact coverage**. This article fit each window and used none of those longer-article transformations. The laptop has an i7-1195G7, about 15 GiB RAM and Intel graphics with no CUDA; all four run on CPU, with the larger two requiring several GB and noticeable waits. Actual latency depends on what else is in memory; the earlier desktop GPU timings are not laptop timings.

The existing [50-article ROUGE comparison](../BENCHMARK_RESULTS.md) gives full-article ROUGE-1 F1 of **40.95 LSTM, 38.75 Transformer, 45.21 BART and 43.81 PEGASUS**. That metric averages overlap with reference summaries on a different, small CNN/DailyMail sample. It does not score whether this fictional article's advisory status is conveyed, and a true but secondary sentence can earn overlap while the main event is missed. BART leading this sample and PEGASUS being preferable on this article can both be true. No benchmark was rerun.

## Next steps, in benefit/effort order

1. **For the class demo, read the model name and summary together after each request finishes; clear or switch models only after completion.** Use the main-event sentence as a human check. This is a presentation practice, not a code change.
2. **If the duplicate display recurs, capture a screenshot plus the browser's `/summarize` request model value and response text, and note the time.** Compare them with the server's model log. This would distinguish a wrong submitted selection, stale browser rendering, and copying the wrong output. Do this before changing cache or UI code.
3. **Review a small, separate set of articles for the main event, resolution and supporting evidence.** Include this case as a diagnostic, but do not tune or choose a default from it alone. Only after that review consider a targeted, reversible change to presentation or generation and evaluate it separately from the frozen 50-article results.

## Exact case evidence

**Supplied article** (title shown for identification; the body alone was inferred as model input):

> Riverton Restores Water Service After Repairs
>
> Riverton officials lifted a boil-water advisory Friday morning after laboratory tests confirmed that the town’s drinking water met safety standards. The advisory had affected approximately 18,000 residents since Tuesday.
>
> The disruption began when a control valve failed during maintenance at the town’s main pumping station. Water pressure dropped across several neighborhoods, creating a risk that contaminants could enter the distribution system. Officials said the advisory was precautionary; contamination had not been confirmed.
>
> Repair crews replaced the damaged valve Tuesday evening. Workers then flushed the affected pipelines and collected water samples from 12 locations. Two consecutive rounds of testing found no harmful bacteria, allowing the town to lift the advisory.
>
> “We understand how disruptive this has been, especially for families and small businesses,” said water department manager Elena Torres. “The repairs are complete, and the test results show that residents can use their tap water normally again.”
>
> During the advisory, residents were instructed to boil water before drinking it, cooking, brushing their teeth, or making ice. Two schools closed Wednesday because they could not provide sufficient drinking water, but reopened Thursday after bottled supplies arrived. The local hospital remained open throughout the disruption.
>
> The emergency response cost approximately $86,000, including repairs, laboratory testing, and bottled-water distribution. The town will cover the expense through its emergency reserve. Officials said no immediate water-rate increase is planned.
>
> Mayor Priya Shah ordered a review of the town’s notification procedures after some residents reported receiving the warning several hours after the pressure loss. The review will examine whether text alerts and coordination with schools can be improved.
>
> A separate proposal to modernize the pumping station remains under consideration. The proposed $3.2 million project has not received funding or final approval.
>
> Health officials said no illnesses had been linked to the incident as of Friday. Residents were advised to run cold taps for five minutes and discard ice made during the advisory.

**Reported displayed LSTM and Transformer, exactly identical:**

> the advisory had affected approximately 18,000 residents since tuesday. water pressure dropped across several neighborhoods, creating a risk that contaminants could enter the distribution system. the town will cover the expense through its emergency reserve.

**Reproduced LSTM direct, pipeline and fresh Gradio route:**

> the advisory had affected 18,000 residents since tuesday. water pressure dropped across several neighborhoods, creating a risk that contaminants could enter the distribution system. two schools closed wednesday because they could not provide sufficient drinking water, but reopened thursday after bottled supplies arrived.

**Reported and reproduced BART:**

> A control valve failed during maintenance at the town’s main pumping station. Water pressure dropped across several neighborhoods, creating a risk that contaminants could enter the distribution system. Officials said the advisory was precautionary; contamination had not been confirmed. Two consecutive rounds of testing found no harmful bacteria.

**Reported and reproduced PEGASUS display:**

> Riverton officials lifted a boil-water advisory Friday morning. The advisory had affected approximately 18,000 residents since Tuesday. The disruption began when a control valve failed during maintenance at the town’s main pumping station.

The raw OpenNMT outputs contain `<t>`/`</t>` around the sentences. LSTM raw tokens omit “approximately” and end with the schools sentence; Transformer raw tokens include “approximately” and end with the emergency-reserve sentence. BART's raw text is the quoted output between `<s>`/`</s>`. PEGASUS's raw text has `<n>` between the three quoted sentences and `</s>` at the end. Those are the only display-relevant cleanups observed here.
