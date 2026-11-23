# Comparing Neural Models for News Summarization

Biruk Anley and Daniela Ortega · 15:00 including the screen switch and demonstration. Sixteen main slides and fifteen appendix slides. Main results use the supplied final training scores on the same 250 full articles.

## 1 · Comparing Neural Models for News Summarization

**Biruk · 00:00–00:45**

If yesterday’s news said to boil your tap water, what would you want to know today? You could read a perfectly accurate sentence about laboratory tests and still have no idea whether the warning is over. That difference between a true detail and the news a reader needs runs through our project. We compared released news summarizers and built a local application to use them. The moving words here compress our fictional Riverton story. They illustrate the task. Later we’ll see what the systems actually wrote.

## 2 · What survives compression?

**Biruk · 00:45–01:25**

We started with two different ways of modeling sequences. One carries a recurrent state from word to word. The other lets positions attend to one another. We wanted to see both in a working application, using the same articles and reference highlights. We also wanted a simple comparison anyone could reproduce by reading the start of a news story. Lead-3 takes the first three sentences. Our work includes model integration, full-article processing, training and evaluation. The final results use our supplied training scores on the same 250 articles.

## 3 · One article, one set of highlights

**Biruk · 01:25–02:20**

CNN/DailyMail pairs news stories with short reference highlights. Here the article attributes Brady’s absence to prior family commitments. The highlight compresses the attribution and names the meeting with Obama. They express a similar point using different words. These are exact excerpts of a saved test row, including the reference’s punctuation spacing. The counts at the top describe version 3.0.0. Training, validation and test splits serve different purposes. We drew our frozen 250 from the test split. Each released checkpoint has its own data history.

## 4 · From article to model input

**Biruk · 02:20–03:20**

The top line describes historical checkpoint preparation using Stanford PTB, lowercasing and a 400-source/100-target convention. It is a release recipe. The large path shows our current app. Typographic quote normalization comes before NLTK Treebank tokenization. **Advance.** The word tokens shown are saved from the earlier Riverton fixture. **Advance.** We load the checkpoint’s existing vocabulary. The numeric word IDs here are illustrative. A separate source dictionary supports copying. **Advance.** The OpenNMT inputter uses torchtext fields and a batch of one with source lengths and the source map. BART instead uses the exact saved subword ID prefix shown below and an all-ones mask for this unpadded single input. PEGASUS has its own SentencePiece tokenizer. **Advance.** Evaluation checks frozen article/reference IDs and text hashes. Original highlights remain the scoring target. Porter stemming happens inside ROUGE. Hand over to Daniela.

## 5 · LSTM

**Daniela · 03:20–04:25**

**Advance through the first three tokens.** An embedding gives each word a vector. Adjacent encoder states depend on one another. There is a forward scan and a reverse scan. Together they give each position left and right context. Within each recurrent direction, the cell state carries memory; the hidden state exposes information to the next step. The bridge initializes the decoder’s hidden and cell state. **Advance.** The decoder writes in one direction, using the previous word and attended output. Its query compares against encoded positions. A weighted sum returns context about the source. **Advance.** The generator blends vocabulary prediction and source copying through a learned gate. Every new token extends the prefix for the next step. The drawn links illustrate dependencies. We did not record these activations.

## 6 · LSTM in practice

**Daniela · 04:25–05:05**

At roughly 43 million parameters this is our smallest checkpoint. Copying gives a name or number a route from source to summary, and recurrence carries information between neighbors. Each step waits for its predecessor, and information about distant words must travel across those steps. The saved LSTM output retained a population figure and the reopening of two schools. Both are supported details. A resident reading only that summary would still have to look up whether the advisory ended.

## 7 · Transformer

**Daniela · 05:05–06:10**

**Advance for embeddings, then source links.** Positions tell the encoder where tokens occur. In each layer, every source position can consult other positions. Heads can form different connections. The feed-forward part transforms each resulting representation. **Advance.** Inside the decoder, masked self-attention can see the available summary prefix. **Advance.** Cross-attention uses the decoder query and the encoder’s keys and values. This asks which source information helps produce the next word. **Advance.** This OpenNMT release reuses its final cross-attention for copying and mixes it with vocabulary prediction. There are four layers on each side. Source representations have parallel work within a layer. Generating the summary still takes one new token at a time.

## 8 · Transformer in practice

**Daniela · 06:10–06:55**

A source word can connect directly with something far away, such as the subject of an action. Dense attention compares positions pairwise, so longer inputs cost memory and computation. Decoder generation also depends on previously chosen words. Those properties explain the architecture. The outputs depend on the particular trained system, too. On Riverton this Transformer kept pressure loss and emergency funding, but omitted the lifting. Now Biruk will show how an article gets to either checkpoint.

## 9 · Where a long article goes

**Biruk · 06:55–08:05**

The user supplies an article and chooses a model. Its tokenizer determines whether the input fits. **Advance.** Fitting articles use one pass. For a longer article the planner groups original spans around sentence boundaries. It can split a sentence that exceeds the window. **Advance.** Each input is summarized with the selected checkpoint. **Advance.** The app merges in source order and deduplicates exact sentences. A merged result that fits can be displayed immediately. If it exceeds capacity, bounded consolidation replans it and checks that the reduction makes progress. Workload guards can return a clear failure. The UI displays the summary, word counts and request statistics. Checking that every source span was processed cannot tell us that the output contains every important fact.

## 10 · Evaluation on 250 articles

**Biruk · 08:05–09:00**

We evaluate the same 250 test articles with full original inputs and reference highlights. ROUGE-1 measures unigram overlap and ROUGE-2 adjacent word pairs. ROUGE-L finds an in-order subsequence, allowing gaps. ROUGE-Lsum uses sentence-aware union matching. We present F1 times 100. A valid paraphrase can receive less overlap, and shared vocabulary can survive an important omission. Lead-3 extracts the first three sentences and provides a useful news baseline. Earlier repository benchmark tables remain in the appendix for historical context.

## 11 · LSTM, Transformer and Lead-3

**Biruk · 09:00–09:45**

Here are the final full-article scores for our core models and baseline. LSTM reached 42.13 on ROUGE-1. **Advance.** Transformer reached 43.47. **Advance.** Lead-3 reached 38.72. **Advance.** Both neural systems exceed Lead-3 across all four ROUGE measures. Transformer also exceeds LSTM on each measure. The columns distinguish unigram overlap, bigram overlap, sequence LCS and sentence-aware LCS. Lead-3 is a useful baseline because news often puts central facts early. These results show what the trained systems achieved on the same 250 articles.

## 12 · What did the larger models learn first?

**Biruk · 09:45–10:40**

BART learned to reconstruct text after corruption. That broad denoising objective came before CNN/DailyMail fine-tuning. PEGASUS removed selected sentences and trained the system to predict them, an objective tied to selecting salient content. The released checkpoint uses mixed pretraining on C4 and HugeNews. BART has about 406 million learned parameters and PEGASUS about 569 million. These are larger systems with different histories, tokenizers and search profiles. Daniela will add them to exactly the same comparison.

## 13 · The same articles, two more systems

**Daniela · 10:40–11:35**

The core rows remain in place. **Advance twice to add BART and PEGASUS.** The axis still runs from zero to fifty. BART scores 46.18 on ROUGE-1 and PEGASUS scores 46.83. **Advance.** PEGASUS is the best-performing model across all four reported ROUGE measures, with 23.76 on ROUGE-2, 33.91 on ROUGE-L and 43.85 on ROUGE-Lsum. Every neural model beats Lead-3 on every metric. PEGASUS leads this comparison, followed by BART, Transformer and LSTM. These measures describe overlap with the reference highlights. Source-based reading adds a check on the events and supporting evidence preserved in an output.

## 14 · Live Demo

**Daniela · 11:35–13:30**

The demo and saved fallbacks illustrate the existing app checkpoints. Switch to the prepared Gradio screen on port 7860. Paste the saved 315-word article body using the checklist below. Read the selected model aloud before submitting one classroom request. Point out the summary, input/output word counts and selected model. Ask the audience to find the lifted advisory and supporting tests or repairs. Allow about 30 seconds of visible CPU waiting within this slot. If the app is unavailable or still working, use the saved BART and PEGASUS outputs on slides 27 and 28 or in the local text file. The reproduced BART output contains negative testing results while omitting the lifting. PEGASUS gives the lifting while omitting repair and testing evidence. These saved examples explain what a source-based check adds. Return to slide 15 to discuss the findings.

## 15 · What we learned

**Daniela · 13:30–14:10**

PEGASUS performs best on all four ROUGE measures in our final comparison. Every neural model exceeds Lead-3 on each measure. These are the central comparative findings from the 250-article full-input evaluation. Source-based review asks further questions. Does a summary identify the central event, who acted, and the evidence behind it? Fluent text can omit the main event. Content selection can fail even with attention and copying. Processing every input span can still lose important facts. Our proposed next step is structured human review on fresh articles.

## 16 · What we tried—and why we changed it

**Daniela · 14:10–15:00**

**Advance once per row.** We implemented Short, Medium, Detailed and percentage targets. Early EOS and similar outputs made lengths unreliable, so the demo now uses natural checkpoint stopping. Repeated consolidation could compress away useful content, so a fitting merge returns directly. The historical 400-token convention caused unnecessary chunks when treated as a hard capacity, so the app uses a guarded 1,024-token work window. Shared decoding overrides changed native behavior. We restored model-specific settings where saved diagnostics supported the change. These are documented engineering decisions. Thank you.

## Offline demo checklist and fallback

- Before class, open the browser presentation on port 8008 and the cached Gradio app on port 7860 in a separate window or screen. When Gradio is already responding, use the existing instance.
- On the prepared Ubuntu laptop, the documented launch from the repository root is `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python app.py --port 7860`. On the prepared Windows PC, set `$env:HF_HUB_OFFLINE='1'` and `$env:TRANSFORMERS_OFFLINE='1'` on separate PowerShell lines, then run `.\.venv\Scripts\python.exe app.py --port 7860`. Open `http://127.0.0.1:7860/`. A fresh clone needs the existing model cache and application setup for a live request.
- Keep the article body and saved text locally accessible. Opening the presentation or PDF never submits a request. [Article body](riverton_article.txt) · [Complete saved outputs](saved_demo_outputs.md)
- During the timed switch, use at most one live request. Fallback slides 25, 26, 27 and 28 contain LSTM, Transformer, BART and PEGASUS respectively. Return directly to the findings at `http://127.0.0.1:8008/#/conclusion`.
- If asked about the earlier identical displays, the reported strings matched the Transformer. A new direct run, pipeline call and Gradio LSTM call produced distinct LSTM text. The original display’s cause remains unresolved.

## Backup discussion

Slide 17 contains the final training results. Slides 18–19 preserve the earlier full-article and shared-prefix benchmarks. Slides 20–22 cover the application checkpoint identity, architecture and search. Slides 23–24 trace length controls and chunking decisions. Slides 25–29 preserve the reproduced strings and duplicate-display diagnosis. Slides 30–31 identify the cited papers and project evidence. [Detailed citations and links](REFERENCES.md)
