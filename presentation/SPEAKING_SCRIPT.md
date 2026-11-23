# Speaking script

Biruk Anley and Daniela Ortega. Sixteen main slides, 15:00 including transitions and the brief demo.

References and backups follow the timed talk. Slide 14 receives 100 seconds, including up to about 30 seconds of CPU wait.

## Slide 1 · Comparing Neural Models for News Summarization

Biruk Anley · 00:00–00:45 · 45 seconds

Imagine opening a news summary to find out whether you can use your tap water again. A summary might correctly mention a broken valve and testing, yet leave you unsure whether the warning ended. Our example compresses a fictional Riverton article into its main event and the evidence behind it. That is the problem we explored. We are Biruk Anley and Daniela Ortega, and our project compares neural news summarizers while building a working interface for complete articles. We will look at how two smaller models work, how we evaluated them, and why reading the source still matters.

**Stage direction**

Point to the lifted advisory before introducing model names.

## Slide 2 · Our target and goals

Biruk Anley · 00:45–01:25 · 40 seconds

We had three goals. First, compare an LSTM system with an encoder–decoder Transformer. Second, make them usable through an article-summarization interface. Third, evaluate returned summaries against saved reference highlights and a simple baseline. Lead-3 takes the first three sentences of an article, which is a useful standard for news. We integrated released pretrained checkpoints. We did the application engineering and evaluation, and we did not train or fine-tune these checkpoint weights. Later we will introduce two larger released systems to put the core comparison in context.

## Slide 3 · CNN/DailyMail pairs articles with highlights

Biruk Anley · 01:25–02:20 · 55 seconds

CNN/DailyMail pairs a news article with short reference highlights. The card for version 3.0.0 lists these training, validation and test splits. Training is used to fit a model, validation helps development, and the test split supports evaluation. The example below is an actual saved row from our fixed test sample. The article reports why Brady missed the White House visit, and the reference captures that point. We evaluated the same 50 test articles for every system, using the frozen seed-42 selection. The released checkpoints have different histories. The OpenNMT releases were trained for CNN/DailyMail summarization. BART and PEGASUS also had broad pretraining before task fine-tuning. These counts describe our dataset version. They do not establish identical historical training data or preprocessing for all checkpoints.

## Slide 4 · From article text to model input

Biruk Anley · 02:20–03:20 · 60 seconds

Before either architecture runs, the software checks that the input is usable and within application workload limits. For OpenNMT, our adapter normalizes curly quotes, splits simple sentence segments, applies a Treebank-style word tokenizer and lowercases its tokens. The line shown contains actual saved tokens from an earlier fictional Riverton fixture. The source starts with Riverton officials lifted a three-day boil-water advisory. The saved vocabulary maps words to model IDs. A source-specific dictionary also lets the generator copy words from the article. Summary decoding starts with a beginning token and can end with an end-of-sequence token. Training used an older Stanford preprocessing convention. Our current NLTK approximation is a documented difference. Sentence boundaries also help plan chunks for long articles. BART and PEGASUS later use their own case-preserving subword tokenizers. Porter stemming will appear in ROUGE scoring.

## Slide 5 · LSTM encoder–decoder with attention

Daniela Ortega · 03:20–04:25 · 65 seconds

An LSTM carries a memory state as it reads or writes a sequence. Here word embeddings turn input words into vectors. The bidirectional encoder reads the article in both directions, so each encoded word can include left and right context. The decoder writes the summary in one direction, carrying its recurrent state from one generated word to the next. Its state queries attention over encoder outputs. Attention supplies a weighted source context. The generator combines vocabulary prediction with copying, using source attention and a learned gate. For example, it may generate lifted from its vocabulary and copy Riverton from the article. Copying can help reproduce names, while deciding which event matters remains the model’s task. This released checkpoint has about 43 million parameters. The arrows show dependencies within a decoding step, and the recurrent loop carries information into the next step.

**Stage direction**

Trace source memory, decoder query, context and copy weights.

## Slide 6 · What the LSTM gives us

Daniela Ortega · 04:25–05:05 · 40 seconds

The LSTM is our smallest checkpoint, making its weight storage relatively modest. Its attention and copy mechanism give it a direct route back to source words, useful for names and numbers. Recurrent computation still proceeds step by step, and carrying context across a long article can be difficult. Those are architectural tendencies. Our saved Riverton observation is more specific. The reproduced LSTM summary retained the population, pressure risk and school details, while omitting that the advisory ended. Copying supported words did not guarantee selecting the important event. One diagnostic example cannot tell us how often that failure occurs.

## Slide 7 · Our encoder–decoder Transformer

Daniela Ortega · 05:05–06:10 · 65 seconds

Our Transformer also has an encoder and a decoder, with four layers in each. Embeddings represent tokens, and positions provide order information. Encoder self-attention lets source tokens consult other source tokens, followed by a feed-forward transformation. In the decoder, masked self-attention sees only the summary words available so far. Cross-attention connects that decoder representation to the encoded article. Think of a query as what the decoder needs, keys as source addresses to compare it with, and values as the information it retrieves. The large horizontal arrow is that source-to-decoder connection. This particular OpenNMT checkpoint also has a vocabulary and copy mixture. It uses its final cross-attention weights for copying. The resulting distribution chooses the next summary word, then generation continues. It has about 81 million parameters.

**Stage direction**

Emphasize the source arrow entering cross-attention.

## Slide 8 · Connections help, but selection still matters

Daniela Ortega · 06:10–06:55 · 45 seconds

Self-attention gives words a way to connect across the source. That helps represent context, and during training many token positions can be computed in parallel. The cost is attention memory that grows with input length. Summary inference still generates one new token after another. These are general architectural properties. Our particular released checkpoint still omitted the lifting in Riverton, and separate saved development reviews found relationship and entity errors. A connection mechanism gives the model access to information. It does not ensure that it selects the central event or preserves who did what. Our comparisons also differ in training, tokenizer and decoding, so they do not isolate architecture alone.

## Slide 9 · How the complete application works

Biruk Anley · 06:55–08:05 · 70 seconds

This is a screenshot of our current local interface. The user supplies an article and chooses a model. The system checks the article using that model’s tokenizer. If the input fits, it gets one direct source pass. Otherwise the planner groups text near sentence boundaries, splitting an oversized sentence safely when needed. Each chunk is summarized. The app joins those outputs in source order and removes exact complete-sentence duplicates. If the merged result fits the model window, it can be returned directly. If further reduction is necessary, the app uses bounded consolidation with limits on passes and total work. These choices try to avoid preventable information loss from unnecessary chunking or repeated compression. Statistics describe the request, including output words and time. Complete input processing leaves a separate question. Did the summary preserve the main event and its evidence? The current interface uses natural checkpoint generation and has a model selector without a summary-length selector.

## Slide 10 · How we measured reference overlap

Biruk Anley · 08:05–09:00 · 55 seconds

We used the same frozen 50 test articles and original highlights for every system. ROUGE compares the model summary with a reference. In this illustrative sentence pair, lifted the advisory appears in both texts. ROUGE-1 counts individual word overlap, while ROUGE-2 compares adjacent pairs. ROUGE-L looks for a matching sequence in order, allowing gaps. ROUGE-Lsum uses sentence-aware matching across the summary. We report F1, which balances precision and recall, multiplied by 100. Porter stemming reduces words to related forms during evaluation. A summary can earn overlap while missing the main event, and a valid paraphrase can receive less overlap. We therefore use ROUGE as one measured comparison, alongside source-grounded reading. The full-article track is our main narrative because it evaluates the application’s handling of complete source articles.

## Slide 11 · Core models and a simple baseline

Biruk Anley · 09:00–09:45 · 45 seconds

Here are the core results from the saved full-article track. LSTM scored 40.95 on ROUGE-1, Transformer 38.75, and Lead-3 40.36. LSTM led this Transformer checkpoint on the three measures shown. Lead-3 is a serious comparison because news often puts central facts early. It exceeded Transformer on these measures while requiring no generated summary. This describes whole systems on a small sample. It does not show that recurrent models are generally better than Transformers. The full four-metric table is in backup, including the separate ROUGE-L values. These are the existing benchmark results, with no new evaluation for this presentation.

## Slide 12 · Larger released comparison systems

Biruk Anley · 09:45–10:40 · 55 seconds

We also integrated two larger released comparison systems. BART-large-CNN has about 406 million learned parameters. Its denoising pretraining teaches it to reconstruct corrupted text, followed by fine-tuning for CNN/DailyMail summarization. PEGASUS-CNN/DailyMail has about 569 million learned parameters. Its gap-sentence objective removes selected sentences and asks the model to generate them. The released card documents mixed pretraining on C4 and HugeNews before task fine-tuning. The size bars use a common scale. These systems bring different training histories as well as more parameters. BART uses byte-level subwords and PEGASUS uses SentencePiece. Both preserve case. They generate through vocabulary distributions, while the OpenNMT checkpoints have an explicit copy gate. We are adding comparison systems to the story using results already saved.

## Slide 13 · The expanded comparison

Daniela Ortega · 10:40–11:35 · 55 seconds

This reveals the remaining systems from the same frozen comparison. We used the same 50 articles, references and saved results. BART led ROUGE-1, ROUGE-2 and ROUGE-Lsum on this sample. Its ROUGE-1 was 45.21, followed by PEGASUS at 43.81. PEGASUS narrowly led the separate flat ROUGE-L metric, 30.92 to BART’s 30.88. All five systems completed 50 requests in each of the two tracks, giving 500 completed evaluations and zero failed requests. Completion means an output was returned under the protocol. These measurements do not establish a general factual-quality winner or statistical significance. We also leave speed out because desktop and game activity contaminated the saved GPU timing context. Next, a diagnostic example shows why an aggregate overlap score cannot answer every useful question.

## Slide 14 · What did the summary leave out?

Daniela Ortega · 11:35–13:15 · 100 seconds

The main event in our latest fictional Riverton article is that officials lifted the boil-water advisory. Repairs, flushing and two clear testing rounds support that decision. The saved BART output includes supported background, including the testing sentence shown, but its full summary omits that the advisory ended. PEGASUS clearly captures the lifting, while omitting the repair and testing evidence. BART gives us a small all-the-details-except-the-headline moment. More precisely, it selected true secondary facts and missed the resolution. Both outputs ended naturally after processing the entire article in one pass. Cleanup or chunking did not remove the lifting from BART. For the brief demo, we will paste the saved article body, check the selected model and submit at most one request. While it runs, compare the returned summary with the source event. If the CPU takes about 30 seconds, we will continue with the complete saved outputs in backup. Check event, resolution and evidence separately. This example does not establish a general model ranking.

**Stage direction**

Reserve about 30 seconds for CPU wait within this 100-second allocation. Submit at most one live request in class. If it is slow, use the saved output slides.

## Slide 15 · Experiments and lessons

Daniela Ortega · 13:15–14:20 · 65 seconds

We tried several engineering choices before settling on the current demo. Short, Medium and Detailed preferences, including percentage targets, were technically implementable. Actual lengths were unreliable. Models could stop early, different choices could produce similar summaries, and forcing more words did not reliably improve useful content. We removed the selector for a simpler, more predictable interface. The older API remains available for experiments. We also changed chunking and consolidation so a fitting article takes one source pass and a fitting merge avoids another compression step. Restoring model-specific search settings recovered the main claim in one saved development case, while other errors remained. Finally, we revisited an input-window assumption. The OpenNMT recipe’s 400 tokens was a training convention. The current application uses a guarded 1024-token work window, with no claim that this extends a learned quality boundary. These are documented decisions and observations, rather than a measured improvement on the frozen test sample.

## Slide 16 · What we learned

Daniela Ortega · 14:20–15:00 · 40 seconds

We built an article interface and compared four released summarizers with Lead-3 on the same fixed articles. BART led most overlap metrics in this small sample. The diagnostic showed that supported details can survive while the main event disappears. Full input processing and fluent output still leave factual coverage to check. Our next useful step is structured human review of events, relationships and evidence on a separate set of articles. That remains proposed work. Thank you. The backups contain full results, checkpoint details and saved demo outputs.

## Demo checklist for Ubuntu

Use the existing cached setup. Stop an existing app on port 7860 before relaunching. From the repository root, run this as one shell line.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python app.py --port 7860
```

Open http://127.0.0.1:7860/. Paste [the saved body](assets/riverton_article.txt), without the title. Read the model name with its result. Submit at most one live request during the talk. After about 30 seconds, use the complete saved outputs on slides 25–28 and continue. Exact fallback text is also in [saved outputs](assets/saved_demo_outputs.md).

The earlier duplicate LSTM/Transformer display remains unresolved. Fresh direct, pipeline and Gradio checks produced a distinct LSTM output. Describe the saved reproduced outputs. Detailed evidence is in slide notes and [references](REFERENCES.md).

## Guideline and preparation scope

No course guideline was located in the repository or supplied attachment. The deck follows the supplied redesign brief. This redesign captured the current app page without submitting an article. It ran no inference, training or benchmark.
