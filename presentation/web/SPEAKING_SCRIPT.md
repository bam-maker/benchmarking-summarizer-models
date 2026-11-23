# Comparing Neural Models for News Summarization

Biruk Anley and Daniela Ortega · 15:00 including the screen switch and live demonstration. Slides 1–16 are the talk. Slides 17–30 are for questions and fallback. The quoted outputs on slides 24–27 are complete.

## 1 · Comparing Neural Models for News Summarization

**Biruk · 00:00–00:45**

If yesterday’s news said to boil your tap water, what would you want to know today? You could read a perfectly accurate sentence about laboratory tests and still have no idea whether the warning is over. That difference between a true detail and the news a reader needs runs through our project. We compared released news summarizers and built a local application to use them. The moving words here compress our fictional Riverton story. They illustrate the task. Later we’ll see what the systems actually wrote.

## 2 · What survives compression?

**Biruk · 00:45–01:25**

We started with two different ways of modeling sequences. One carries a recurrent state from word to word. The other lets positions attend to one another. We wanted to see both in a working application, using the same articles and reference highlights. We also wanted a simple comparison anyone could reproduce by reading the start of a news story. Lead-3 takes the first three sentences. All the neural systems use released weights. Our work was integration, full-article processing and evaluation.

## 3 · One article, one set of highlights

**Biruk · 01:25–02:20**

CNN/DailyMail pairs news stories with short reference highlights. Here the article attributes Brady’s absence to prior family commitments. The highlight compresses the attribution and names the meeting with Obama. They express a similar point using different words. These are exact excerpts of a saved test row, including the reference’s punctuation spacing. The counts at the top describe version 3.0.0. Training, validation and test splits serve different purposes. We drew our frozen 50 from the test split. Each released checkpoint has its own data history.

## 4 · From article to model-ready input

**Biruk · 02:20–03:20**

The OpenNMT release recipe used Stanford CoreNLP PTB tokenization, lowercasing and a 400-source-token training convention. Today our adapter normalizes typographic quotes, splits simple sentence segments, and uses NLTK Treebank. **Advance.** On an earlier Riverton fixture it produced the lowercase words shown here. **Advance.** The checkpoint already supplies the word vocabulary. The source-specific dictionary maps article positions for copying. **Advance.** BART takes a different route with byte-level BPE. The displayed integers are the actual beginning of its saved input, including the leading special token. The encoder receives IDs and an attention mask. PEGASUS uses its own SentencePiece vocabulary. Both preserve case. For evaluation we kept the selected test IDs and original highlights. Stemming happens in the scorer. Hand over to Daniela.

## 5 · Memory moves one token at a time

**Daniela · 03:20–04:25**

**Advance through the first three tokens.** An embedding gives each word a vector. Adjacent encoder states depend on one another. There is a forward scan and a reverse scan. Together they give each position left and right context. Within each recurrent direction, the cell state carries memory; the hidden state exposes information to the next step. The bridge initializes the decoder’s hidden and cell state. **Advance.** The decoder writes in one direction, using the previous word and attended output. Its query compares against encoded positions. A weighted sum returns context about the source. **Advance.** The generator blends vocabulary prediction and source copying through a learned gate. Every new token extends the prefix for the next step. The drawn links illustrate dependencies. We did not record these activations.

## 6 · LSTM in practice

**Daniela · 04:25–05:05**

At roughly 43 million parameters this is our smallest checkpoint. Copying gives a name or number a route from source to summary, and recurrence carries information between neighbors. Each step waits for its predecessor, and information about distant words must travel across those steps. The saved LSTM output retained a population figure and the reopening of two schools. Both are supported details. A resident reading only that summary would still have to look up whether the advisory ended.

## 7 · Source words can attend in parallel

**Daniela · 05:05–06:10**

**Advance for embeddings, then source links.** Positions tell the encoder where tokens occur. In each layer, every source position can consult other positions. Heads can form different connections. The feed-forward part transforms each resulting representation. **Advance.** Inside the decoder, masked self-attention can see the available summary prefix. **Advance.** Cross-attention uses the decoder query and the encoder’s keys and values. This asks which source information helps produce the next word. **Advance.** This OpenNMT release reuses its final cross-attention for copying and mixes it with vocabulary prediction. There are four layers on each side. Source representations have parallel work within a layer. Generating the summary still takes one new token at a time.

## 8 · Transformer in practice

**Daniela · 06:10–06:55**

A source word can connect directly with something far away, such as the subject of an action. Dense attention compares positions pairwise, so longer inputs cost memory and computation. Decoder generation also depends on previously chosen words. Those properties explain the architecture. The outputs depend on the particular trained system, too. On Riverton this Transformer kept pressure loss and emergency funding, but omitted the lifting. Now Biruk will show how an article gets to either checkpoint.

## 9 · Where a long article goes

**Biruk · 06:55–08:05**

The user supplies an article and chooses a model. Its tokenizer determines whether the input fits. **Advance.** Fitting articles use one pass. For a longer article the planner groups original spans around sentence boundaries. It can split a sentence that exceeds the window. **Advance.** Each input is summarized with the selected checkpoint. **Advance.** The app merges in source order and deduplicates exact sentences. A merged result that fits can be displayed immediately. If it exceeds capacity, bounded consolidation replans it and checks that the reduction makes progress. Workload guards can return a clear failure. The UI displays the summary, word counts and request statistics. Checking that every source span was processed cannot tell us that the output contains every important fact.

## 10 · Two ways to hold the input fixed

**Biruk · 08:05–09:00**

We used the first 50 IDs of our frozen seed-42 test sample and kept the original references in both tracks. Full article measures the end-to-end systems. Shared prefix gives every system the same beginning, bounded by 400 OpenNMT tokens and both subword capacities. The appendix keeps the tracks separate. ROUGE-1 measures unigram overlap and ROUGE-2 adjacent pairs. ROUGE-L finds an in-order subsequence, allowing gaps. ROUGE-Lsum uses sentence-aware union matching. We report F1 times 100, with Porter stemming. A paraphrase can lower overlap, and shared vocabulary can survive an important omission.

## 11 · A simple lead is hard to beat

**Biruk · 09:00–09:45**

Start with unigram overlap. The LSTM scored 40.95. **Advance.** The Transformer scored 38.75. **Advance.** Just taking the first three sentences scored 40.36. **Advance.** Journalists often put important facts early. That makes this cheap baseline a serious point of comparison. These differences describe the particular systems and articles we tested. We did not test statistical significance or isolate architecture as the cause.

## 12 · What did the larger models learn first?

**Biruk · 09:45–10:40**

BART learned to reconstruct text after corruption. That broad denoising objective came before CNN/DailyMail fine-tuning. PEGASUS removed selected sentences and trained the system to predict them, an objective tied to selecting salient content. The released checkpoint uses mixed pretraining on C4 and HugeNews. BART has about 406 million learned parameters and PEGASUS about 569 million. These are larger systems with different histories, tokenizers and search profiles. Daniela will add them to exactly the same comparison.

## 13 · The same articles, two more systems

**Daniela · 10:40–11:35**

The three core rows are already in place, on the same zero-to-fifty scale. **Advance twice to add BART and PEGASUS.** BART reached 45.21 on ROUGE-1 and PEGASUS 43.81. **Advance.** Across all four measures BART led three. PEGASUS narrowly led ROUGE-L, 30.92 to 30.88 as displayed. ROUGE-Lsum measures sentence-aware overlap separately. All systems returned results on these 50 articles under each track. Desktop activity affected the recorded GPU timings, so we leave speed out of this comparison. Let’s ask a question a person could answer by reading.

## 14 · What overlap leaves open

**Daniela · 11:35–12:25**

The source says Riverton lifted its advisory Friday after repairs, flushing and two clear testing rounds. BART reproduces the negative test results. Its full output never tells the reader the warning ended. PEGASUS gives us the headline in its opening sentence but leaves out the repairs and tests. Both ended naturally on a one-pass input. Reading actual outputs lets us separate event, resolution and evidence. This is a single fictional diagnostic article, separate from the 50 benchmark articles.

## 15 · Live Demo

**Daniela · 12:25–14:20 · includes the screen switch**

Switch to the prepared Gradio screen on port 7860. Paste the saved 315-word article body using the checklist below. Read the selected model aloud before submitting one classroom request. Point out its summary, input/output word counts and selected model. Ask the audience to locate the lifting and the evidence for it. Allow about 30 seconds of visible CPU waiting within this slot. If the app is unavailable or still working, use the saved BART and PEGASUS outputs on slides 26 and 27 or in the local text file. Return to slide 16 for the close. Leave the running app independent of the slide browser.

## 16 · Read the summary against the story

**Daniela · 14:20–15:00**

Four released systems and a simple baseline gave us different answers on the same articles. BART led most full-article overlap measures in this small sample. Reading Riverton showed how a summary can preserve true details while omitting the very change a reader wants to know about. Our next proposed evaluation is structured human review of events, relationships and supporting evidence on fresh articles. Thank you.

## Offline demo checklist and fallback

- Before class, open the browser presentation on port 8008 and the cached Gradio app on port 7860 in a separate window or screen. When Gradio is already responding, use the existing instance.
- On the prepared Ubuntu laptop, the documented launch from the repository root is `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python app.py --port 7860`. On the prepared Windows PC, set `$env:HF_HUB_OFFLINE='1'` and `$env:TRANSFORMERS_OFFLINE='1'` on separate PowerShell lines, then run `.\.venv\Scripts\python.exe app.py --port 7860`. Open `http://127.0.0.1:7860/`. A fresh clone needs the existing model cache and application setup for a live request.
- Keep the article body and saved text locally accessible. Opening the presentation or PDF never submits a request. [Article body](riverton_article.txt) · [Complete saved outputs](saved_demo_outputs.md)
- During the timed switch, use at most one live request. Fallback slides 24, 25, 26 and 27 contain LSTM, Transformer, BART and PEGASUS respectively. Return directly to the conclusion at `http://127.0.0.1:8008/#/conclusion`.
- If asked about the earlier identical displays, the reported strings matched the Transformer. A new direct run, pipeline call and Gradio LSTM call produced distinct LSTM text. The original display’s cause remains unresolved.

## Backup discussion

Slides 17–18 contain the two complete metric tables, including both LCS metrics and generation lengths. Slides 19–21 cover checkpoint identity, architecture and search. Slides 22–23 trace length controls and chunking decisions. Slides 24–28 preserve the reproduced strings and the limits of the duplicate-display diagnosis. Slides 29–30 identify the cited papers and the project evidence. [Detailed citations and links](../REFERENCES.md)
