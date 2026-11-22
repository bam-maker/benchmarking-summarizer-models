# Speaking script — Comparing Neural Models for News Summarization

**Presenters:** Biruk Anley and Daniela Ortega · **Course:** CIS 634

**Run time:** 15:00, including transitions and a short live demonstration. References and backups are outside the eight main slides. The stage directions are not spoken. The presentation guideline PDF was not found in the repository or available evidence; the supplied brief governs this presentation.

| Slide | Presenter | Duration | Clock |
|---|---|---:|---|
| 1 | Biruk Anley | 60 s | 0:00–1:00 |
| 2 | Daniela Ortega | 100 s | 1:00–2:40 |
| 3 | Biruk Anley | 130 s | 2:40–4:50 |
| 4 | Daniela Ortega | 100 s | 4:50–6:30 |
| 5 | Biruk Anley | 100 s | 6:30–8:10 |
| 6 | Daniela Ortega | 130 s | 8:10–10:20 |
| 7 | Biruk Anley with Daniela Ortega operating the demo | 190 s | 10:20–13:30 |
| 8 | Daniela Ortega | 90 s | 13:30–15:00 |

## 1. Comparing Neural Models for News Summarization

**Biruk Anley · 60 seconds**

A useful news summary should let a reader understand what happened without reading the entire article. Making text shorter is easy to describe. Deciding which facts must survive is the difficult part.

We are Biruk Anley and Daniela Ortega, and this is our CIS 634 project. We integrated four released summarization checkpoints, built a local application, and evaluated saved outputs against reference summaries. We did not train these released models.

Our two core models are LSTM and Transformer. Later, we add BART and PEGASUS as larger comparison systems. We will explain the application, show our measured results, and finish with an example where a fluent summary misses the headline.

*Stage directions: Show the article-to-summary visual. Pause after “which facts must survive.” Transition at 1:00. The presentation guideline PDF was not found in the repository or available evidence; this deck follows the supplied presentation brief.*

**Evidence for notes:** Repository: README.md; PROJECT_STATUS.md; MODEL_CAPABILITIES.md.

## 2. Two core models learn from article–summary pairs

**Daniela Ortega · 100 seconds**

Our core checkpoints are two ways of learning the same kind of task: read an article and predict a short summary. The training material pairs a news article with reference highlights. During summarization training, the system learns to predict reference tokens from the article and the preceding summary tokens.

CNN/DailyMail supplies news articles and their associated highlights. The small pair on this slide is illustrative: it explains the format rather than claiming to be a retrieved dataset record.

The released OpenNMT LSTM and Transformer checkpoints were trained for CNN/DailyMail summarization. Their documented recipe uses lowercased word tokens, copying, and a historical convention of the first 400 source tokens and up to 100 target tokens. We used the released checkpoints and compatibility adapters; we did not perform their training.

Our evaluation is a separate decision. We used CNN/DailyMail version 3.0.0, selecting 50 test articles from a frozen seed-42 sample. That does not prove that all four released checkpoints used identical historical data preparation or training splits. Keeping model provenance separate from our test selection makes the comparison more honest.

*Stage directions: Point from article to reference highlights. Identify the example as illustrative aloud. Hand off to Biruk at 2:40.*

**Evidence for notes:** Repository: MODEL_CAPABILITIES.md; OpenNMT_Verified_Setup/checkpoint_manifest.json; tests/final_benchmark.py; reports/pilot50/results.json. OpenNMT original summarization recipe: https://opennmt.net/OpenNMT-py/examples/summary/Summarization.html . CNN/DailyMail source/preparation: https://github.com/abisee/cnn-dailymail ; dataset card: https://huggingface.co/datasets/abisee/cnn_dailymail . Exact saved article/reference hashes are authoritative; the retrieval API did not pin a dataset revision.

## 3. Two routes from article to summary

**Biruk Anley · 130 seconds**

Both models turn words into numerical embeddings, encode the article, and generate the summary one token at a time. The diagrams show the actual kinds of components our checkpoints use.

The LSTM encoder reads the article in both directions. Its recurrent state carries information along the sequence. The decoder also maintains a state as it generates each word. Attention lets it consult different encoded article positions instead of relying only on one final article vector. The decoder also receives its previous attention context, which is called input feeding.

Our LSTM has one encoder layer and one decoder layer, 128-dimensional embeddings, and a 512-dimensional combined encoder representation: 256 in each direction. It has about 43 million parameters. The recurrent path is compact, but remembering and selecting relationships across a long article remains difficult.

The Transformer uses position information and self-attention to relate article positions. Its decoder uses masked self-attention over the summary so far, then cross-attention to the encoded article. It has four encoder and four decoder layers, eight attention heads, 512-dimensional states, and about 81 million parameters.

Both checkpoints have an explicit copying mechanism. A learned gate mixes the probability of choosing a vocabulary word with the probability of copying a word from the article. In this Transformer implementation, copying uses the final decoder cross-attention map. Copying helps reproduce source names and wording, but it cannot guarantee that the chosen sentence is the important one. Attention is access to evidence; it is not a fact checker.

*Stage directions: Trace the LSTM route, then Transformer route. Point to both encoder-to-decoder attention connections and both vocabulary/copy mixtures. Do not imply bidirectional generation: only the LSTM encoder reads both ways. Transition at 4:50.*

**Evidence for notes:** Repository: reports/LAPTOP_MODEL_DIAGNOSIS.md; MODEL_CAPABILITIES.md; summarizer.py; OpenNMT_Verified_Setup/legacy/onmt/encoders/rnn_encoder.py; legacy/onmt/decoders/decoder.py; legacy/onmt/decoders/transformer.py; legacy/onmt/modules/copy_generator.py. LSTM: Hochreiter & Schmidhuber (1997), https://doi.org/10.1162/neco.1997.9.8.1735 . Attention: Bahdanau et al., https://arxiv.org/abs/1409.0473 . Transformer: Vaswani et al. (2017), https://papers.nips.cc/paper/7181-attention-is-all-you-need . Copying: See et al. (2017), https://aclanthology.org/P17-1099/ . Our checkpoints have no learned coverage-attention component; a decoding coverage penalty is a separate mechanism.

## 4. A local app handles short and long articles

**Daniela Ortega · 100 seconds**

Our integration makes these models usable through a local Gradio application. Select a model, paste an article, and press Summarize. The result panel shows the summary and compact statistics. There is no summary-length selector in the current interface. This slide is a labeled schematic of the current interface, because we did not have a saved screenshot.

The application first checks whether the article fits the selected model’s token window. A fitting article uses one source pass. A longer article is divided near sentence boundaries, with oversized sentences split safely. The system summarizes each chunk, merges results in source order, removes exact duplicate sentences, and uses bounded consolidation when necessary.

The important distinction is that every source character can be processed while important facts still disappear from the summary. Input coverage is a software property; factual coverage requires reading the output against the source.

We prepared the Ubuntu laptop using its existing CPU path and cached checkpoints. All four models completed offline checks. The first request for a model includes loading time, and the larger comparison models require more memory and patience on this laptop.

*Stage directions: Call out “schematic” visibly and aloud. Trace the fits/long-article branches. Do not launch inference while preparing this presentation. Hand off at 6:30.*

**Evidence for notes:** Repository: app.py; article_summarizer.py; summarizer.py; hf_summarizer.py; PROJECT_STATUS.md; README.md. Current policy: natural-checkpoint-v1, model-specific deterministic search and natural EOS; application work windows are 1024 model tokens, including HF special tokens. CPU laptop: Intel Core i7-1195G7, about 15 GiB RAM, no supported CUDA device.

## 5. Five systems, two evaluation tracks

**Biruk Anley · 100 seconds**

We added two larger released comparison models: BART-large-CNN, with about 406 million learned parameters, and PEGASUS-CNN/DailyMail, with about 569 million. Both were broadly pretrained and then fine-tuned for summarization. BART learns to reconstruct corrupted text; PEGASUS pretraining removes important sentences and learns to generate them. We also include Lead-3, which simply extracts the first three article sentences.

Every system uses the same fixed 50 test articles. The full-article track evaluates our application’s handling of the complete article. The shared-prefix track gives every system identical original text that fits within 400 OpenNMT tokens and both other tokenizers’ capacities. The shared-prefix track still uses the full original reference highlights, which may mention facts beyond that prefix.

We measure ROUGE F1 overlap. ROUGE-1 compares individual words; ROUGE-2 compares word pairs. ROUGE-L uses the longest matching word sequences in order, while ROUGE-Lsum makes a sentence-aware summary-level comparison. Scores are scaled to 100 and use Porter stemming.

These metrics are useful reference comparisons. They do not verify factual accuracy. The larger 1,000-article run was interrupted; the completed study presented here is the 50-article comparison.

*Stage directions: Introduce BART and PEGASUS here, after the core-model explanation. Show separate track cards. Transition at 8:10.*

**Evidence for notes:** Repository: reports/pilot50/results.json; BENCHMARK_RESULTS.md; tests/final_benchmark.py; model_manifest.json; MODEL_CAPABILITIES.md. BART: Lewis et al. (2020), https://aclanthology.org/2020.acl-main.703/ ; checkpoint https://huggingface.co/facebook/bart-large-cnn . PEGASUS: Zhang et al. (2020), https://proceedings.mlr.press/v119/zhang20ae.html ; checkpoint https://huggingface.co/google/pegasus-cnn_dailymail . ROUGE: Lin (2004), https://aclanthology.org/W04-1013/ ; implementation https://github.com/google-research/google-research/tree/master/rouge . PEGASUS 568,699,904 learned parameters; 570,797,056 instantiated includes fixed sinusoidal position tables.

## 6. BART leads most overlap metrics in this sample

**Daniela Ortega · 130 seconds**

This chart shows the full-article track, with the same model order and model colors throughout. We show three metrics here for readability; the complete four-metric tables and shared-prefix results are in the backup slides.

BART has the highest ROUGE-1, ROUGE-2, and ROUGE-Lsum on these 50 articles. Its ROUGE-1 is 45.21, compared with 43.81 for PEGASUS, 40.95 for LSTM, 38.75 for Transformer, and 40.36 for Lead-3. PEGASUS narrowly has the highest flat ROUGE-L: 30.92 versus BART’s 30.88.

Lead-3 is a useful reminder that news articles often place important information early. A learned system needs to be compared with a simple baseline, rather than being impressive only because it generates new text. LSTM and Transformer also change relative order between the two tracks, so the application’s input treatment matters.

All five systems completed all 50 requests in both tracks: 500 evaluations with no failed requests. Successful completion means a summary was returned under the protocol; it does not mean the summary preserved every fact.

We do not claim statistical significance, a general factual-quality winner, or a controlled speed ranking. The recorded desktop GPU timing context included competing activity, so those numbers cannot support a fair speed comparison. Our conclusion is narrower: BART led most reference-overlap metrics in this small, frozen sample. Now we will look at something an aggregate score cannot explain by itself.

*Stage directions: Use full-article results only in the main chart. Read the key pattern, not every bar. Note the narrow PEGASUS ROUGE-L lead, and avoid calling BART the best model overall. Hand off at 10:20.*

**Evidence for notes:** Repository: reports/pilot50/results.json (results.full_article and results.common_input); BENCHMARK_RESULTS.md; PROJECT_STATUS.md. Exact rounded full-article R1/R2/RL/RLsum: LSTM40.95/20.34/28.71/38.22; Transformer38.75/17.35/26.99/36.09; BART45.21/22.50/30.88/42.73; PEGASUS43.81/22.43/30.92/40.91; Lead-3 40.36/18.28/24.91/36.30. No new benchmark was run for the presentation.

## 7. A good score can still miss the headline

**Biruk Anley with Daniela Ortega operating the demo · 190 seconds**

Riverton is a fictional diagnostic article, separate from the CNN/DailyMail benchmark. Its main event is clear: officials lifted the boil-water advisory Friday after completed repairs and two clear testing rounds.

BART’s saved summary includes supported background facts and ends, “Two consecutive rounds of testing found no harmful bacteria.” But it never says the advisory was lifted. All the details… except the headline. It does not explicitly claim the warning is still active; it leaves the current status unclear.

PEGASUS begins, “Riverton officials lifted a boil-water advisory Friday morning.” That better captures this example’s main event, although it also omits the completed repairs and testing evidence.

Let us open the prepared local app and use this exact article body. We will submit at most one model if the laptop is responsive, and compare the result with these saved outputs. While it runs, look for the current advisory status, the repair, and both testing rounds.

The earlier displayed LSTM and Transformer texts were identical. Direct checks and a fresh LSTM app request returned a distinct LSTM result. The original browser mismatch remains unresolved; current routing did not reproduce a swap. This example illustrates content selection failures, rather than proving a general model ranking.

*Stage directions: Allocate about90 seconds of this slide to opening the app, pasting the exact315-word body, one optional submission, CPU wait, and inspecting status/repair/testing. Do not generate all four models live. If the visible wait reaches about 30 seconds, preserve the time box and use the saved complete summaries in the backup slides/Markdown appendix. Do not change production decoding. Say when using a saved output. Use body only; log word/character counts support title omission but original request text was not captured. Transition at 13:30.*

**Evidence for notes:** Repository: reports/LAPTOP_MODEL_DIAGNOSIS.md, latest case only. All four direct and pipeline checks used one chunk/one call, full2202-character body, natural EOS before cap. BART omission is present in raw generation, not introduced by UI cleanup. Reported LSTM/Transformer duplicate is an unresolved original-display mismatch; do not call it a confirmed routing bug. Exact body SHA-256648c2a202c8912f6d1fc5bbe468162d2670be52508916bbbf4ac7fd7e5457fa8.

## 8. We built a usable demo; factual review comes next

**Daniela Ortega · 90 seconds**

We built a working local application, integrated four verified released checkpoints, prepared an offline Ubuntu demonstration, and completed a reproducible 50-article comparison with two clearly separated tracks.

Our core LSTM and Transformer models can copy source wording and still miss the main event or confuse relationships. The historical 400-token OpenNMT training convention also differs from the application’s expanded window. Longer articles add chunk-boundary and merging challenges, and attention does not remove those limitations. All four models can omit important facts.

This compares complete systems with different sizes, tokenizers, training histories, and decoding. It cannot isolate architecture alone as the reason for a score difference.

The next useful step is structured human factuality review: check the main event, resolution, entities, relationships, and unsupported claims, ideally with independent reviewers. We should also capture the original request and response if the display mismatch recurs before choosing a fix. The frozen benchmark remains our reference. Our demo is usable; judging its summaries still requires a reader.

*Stage directions: Leave the three conclusions visible. Finish by15:00; references and backups are outside the eight main slides. Invite questions only after the timed talk.*

**Evidence for notes:** Repository: PROJECT_STATUS.md; MODEL_CAPABILITIES.md; BENCHMARK_RESULTS.md; reports/LAPTOP_MODEL_DIAGNOSIS.md. Human rating fields in the saved review packet remain blank; no population factual-error rates or human preference scores have been measured.

## Demo checklist — prepare before class

1. Open a terminal at the repository root. Stop any previous app on port 7860 before relaunching. Reuse the prepared environment and cached checkpoints.
2. Launch with the exact documented offline command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python app.py --port 7860
```

3. Open <http://127.0.0.1:7860/>. Leave the app ready before the talk. The first request for a model includes loading, and CPU waits depend on available memory and other activity.
4. Copy the exact article **body** below. Its title is for identification and should not be pasted into the demo input. The saved diagnosis inferred body-only input from the matching 315-word/2,202-character console counts.
5. On slide 7, select at most one model, read the model name aloud, press Summarize once, and wait for completion. PEGASUS is a useful choice for discussing the main-event sentence. Keep the complete saved outputs available and identify them as saved evidence when used.
6. If the model is still loading or generating after the allotted wait, continue with the saved evidence. Do not depend on four live generations within the talk. Check the advisory status, completed repairs, and two testing rounds; technical success does not certify those facts.

No model inference, training, benchmark, or application environment change was performed to write these notes.

## Exact latest fictional diagnostic article — demo input

**Identification title, not part of the reproduced body:** Riverton Restores Water Service After Repairs

Riverton officials lifted a boil-water advisory Friday morning after laboratory tests confirmed that the town’s drinking water met safety standards. The advisory had affected approximately 18,000 residents since Tuesday.

The disruption began when a control valve failed during maintenance at the town’s main pumping station. Water pressure dropped across several neighborhoods, creating a risk that contaminants could enter the distribution system. Officials said the advisory was precautionary; contamination had not been confirmed.

Repair crews replaced the damaged valve Tuesday evening. Workers then flushed the affected pipelines and collected water samples from 12 locations. Two consecutive rounds of testing found no harmful bacteria, allowing the town to lift the advisory.

“We understand how disruptive this has been, especially for families and small businesses,” said water department manager Elena Torres. “The repairs are complete, and the test results show that residents can use their tap water normally again.”

During the advisory, residents were instructed to boil water before drinking it, cooking, brushing their teeth, or making ice. Two schools closed Wednesday because they could not provide sufficient drinking water, but reopened Thursday after bottled supplies arrived. The local hospital remained open throughout the disruption.

The emergency response cost approximately $86,000, including repairs, laboratory testing, and bottled-water distribution. The town will cover the expense through its emergency reserve. Officials said no immediate water-rate increase is planned.

Mayor Priya Shah ordered a review of the town’s notification procedures after some residents reported receiving the warning several hours after the pressure loss. The review will examine whether text alerts and coordination with schools can be improved.

A separate proposal to modernize the pumping station remains under consideration. The proposed $3.2 million project has not received funding or final approval.

Health officials said no illnesses had been linked to the incident as of Friday. Residents were advised to run cold taps for five minutes and discard ice made during the advisory.

## Saved outputs — fallback, not a new run

These are exact latest saved displayed summaries from [the laptop diagnosis](../reports/LAPTOP_MODEL_DIAGNOSIS.md). The LSTM text below is the distinct reproduced result, rather than the originally duplicated display.

### LSTM — reproduced direct, pipeline, and fresh Gradio route

> the advisory had affected 18,000 residents since tuesday. water pressure dropped across several neighborhoods, creating a risk that contaminants could enter the distribution system. two schools closed wednesday because they could not provide sufficient drinking water, but reopened thursday after bottled supplies arrived.

### Transformer — reported and reproduced

> the advisory had affected approximately 18,000 residents since tuesday. water pressure dropped across several neighborhoods, creating a risk that contaminants could enter the distribution system. the town will cover the expense through its emergency reserve.

### BART — reported and reproduced

> A control valve failed during maintenance at the town’s main pumping station. Water pressure dropped across several neighborhoods, creating a risk that contaminants could enter the distribution system. Officials said the advisory was precautionary; contamination had not been confirmed. Two consecutive rounds of testing found no harmful bacteria.

### PEGASUS — reported and reproduced display

> Riverton officials lifted a boil-water advisory Friday morning. The advisory had affected approximately 18,000 residents since Tuesday. The disruption began when a control valve failed during maintenance at the town’s main pumping station.

## Backup talking points

- **Full tables:** keep full-article and shared-prefix metrics separate. Every value is ROUGE F1 ×100; sample size is50 per system per track. Lead-3 is extraction of the first three sentences. No statistical-significance or controlled-speed claim is supported.
- **Checkpoint details:** OpenNMT model identities, original SHA-256 hashes, and pinned legacy commit are in [the checkpoint manifest](../OpenNMT_Verified_Setup/checkpoint_manifest.json). BART and PEGASUS IDs/revisions are in [the HF manifest](../model_manifest.json). We integrated released checkpoints; we did not train them.
- **Decoding:** current app uses natural-checkpoint-v1. OpenNMT beam5 with length/coverage penalties and trigram blocking; BART beam4, length penalty2, trigram blocking; PEGASUS beam8, length penalty0.8, no n-gram blocking. Native minimums are bounded for the source, forced HF EOS is disabled, and retries are bounded. See [current status](../PROJECT_STATUS.md) and [latest diagnosis](../reports/LAPTOP_MODEL_DIAGNOSIS.md) for exact case settings.
- **Input windows:**1024 model tokens per application pass, with model-specific tokenizers; this differs from the historical OpenNMT first400-token convention. More input processed is not proof of better factual coverage.
- **Duplicate display question:** the original two displayed strings matched exactly. Direct LSTM raw tokens differed, pipeline cleanup did not collapse them, and a fresh live LSTM route returned the distinct LSTM result. Original browser selection/response was not captured; the cause remains unresolved.
- **Provenance warning:** the older Riverton fixture in reports/RIVERTON_DIAGNOSIS.md is a different article. Its PEGASUS omission is not the latest classroom example. Do not mix them.
- **Human review:** no completed human factuality ratings, training curves, epochs from our work, training losses, or population error rates are available. Future work is proposed rather than completed.
