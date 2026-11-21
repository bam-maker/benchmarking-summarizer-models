# Verified CNN/DailyMail LSTM and Transformer setup

Checked November 20, 2026. This is an inference compatibility kit, not the finished class project or web app. No models were trained or fine-tuned.

## Verified result

Both official checkpoint downloads completed, loaded strictly into the model architecture, and generated summaries on CPU. Both ran the Short, Medium, and Detailed decoding presets on one synthetic English news article. Repetition blocking was enabled. These are smoke tests, not a dataset quality evaluation.

| Model | Actual unique parameters | Architecture saved in checkpoint |
|---|---:|---|
| LSTM | 42,789,205 | One-layer bidirectional LSTM encoder and one-layer LSTM decoder; 128-dimensional embeddings; hidden size 512; additive attention; copying; no learned coverage attention |
| Transformer | 81,468,757 | Four encoder and four decoder layers; 512-dimensional representations; shared source/target embeddings; positional encoding; copying |

The saved training options have no pretrained encoder or decoder word vectors. The original OpenNMT documentation describes direct CNN/DailyMail training, rather than a language-pretraining/fine-tuning pipeline. This inspection does not independently audit the original training dataset or data leakage.

Documentation and checkpoint provenance: https://opennmt.net/OpenNMT-py/examples/summary/Summarization.html

## Tested environment

- Linux x86_64; CPU only; Python 3.11.18.
- PyTorch 2.5.1+cpu; torchtext 0.6.0; NumPy 1.26.4.
- OpenNMT-py tag 0.2.1, commit `6db7ec1b3afae28c257a0416dd947022c3b12bc0`.
- Six 1.17.0; tqdm 4.70.1; future 1.0.0.
- Two PyTorch CPU threads; beam size four; batch size one.

Windows, GPU execution, Gradio integration, and deployment have not been tested. Use a dedicated environment. Do not install the current OpenNMT package on top of this checkout. On Windows, WSL2 with Ubuntu is the closest match to the tested platform; native Windows compatibility has not been established.

## Reproduce

Prerequisites: Git, Python 3.11, internet access, and approximately 6 GB of free disk space. The checkpoint downloads total approximately 1.48 GB. The script also writes converted inference checkpoints. The archive itself intentionally contains no model weights.

Extract the archive and open a terminal inside `OpenNMT_Verified_Setup`.

On Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

For an experimental native Windows setup, use PowerShell and avoid activation-policy changes by calling the environment interpreter directly:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare.py
.\.venv\Scripts\python.exe smoke_test.py
```

On Linux, after activating the environment:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python prepare.py
python smoke_test.py
```

`prepare.py` clones the pinned official OpenNMT code, applies two compatibility changes, downloads the official checkpoints, verifies their measured SHA-256 hashes, and strips the legacy optimizer state for inference. It does not train. Conversion deserializes the verified official checkpoint files; do not adapt it to arbitrary untrusted checkpoint URLs.

The changes preserve the intended inference behavior: invert a Boolean attention mask with `~mask.bool()` and perform integer beam-index division with `//`. Optimizer objects are read as inert training-state containers during conversion because inference does not use them. Model/generator parameters and vocabularies are retained.

## Smoke-test observations

See `article.txt` for the synthetic input and `smoke_results.json` for all six actual outputs and timings. Timings exclude loading, are measured on this hosted CPU, and should not be used to predict another computer's latency.

| Model | Preset | Actual output words in original check |
|---|---|---:|
| LSTM | Short | 16 |
| LSTM | Medium | 23 |
| LSTM | Detailed | 47 |
| Transformer | Short | 28 |
| Transformer | Medium | 28 |
| Transformer | Detailed | 50 |

Short and Medium produced identical Transformer summaries on this example. This is evidence that decoding budgets do not guarantee distinct content or a target word count. The current presets are diagnostics, not validated product settings. Minimum/maximum token counts include sentence tags and punctuation and are not word budgets. With tighter caps, a generation may end mid-sentence.

An initial LSTM generation without repetition blocking duplicated a sentence. Trigram blocking removed that duplicate in the subsequent smoke test. The outputs also omitted the article's main 200-bus replacement plan, so successful execution is not evidence of complete content selection.

## Next steps for the class project

1. Confirm CPU inference works on your machine before selecting a CUDA build.
2. Keep each checkpoint's vocabulary and reproduce the original tokenization. The synthetic smoke input was manually lowercased and punctuation-tokenized; arbitrary raw article preprocessing is not implemented here.
3. Implement adapters with a common `summarize(article, length_level)` interface.
4. Respect/document the original 400-token source training truncation. Do not silently drop the rest of a long article. Chunking changes the method and must be evaluated.
5. Test length presets on a fixed validation sample, checking repetition, unsupported facts, unfinished sentences, and whether longer summaries add useful content.
6. Build Gradio only after both adapters work; show model, approximate length choice, measured word count, latency, and truncation/termination notices.
7. Use the same held-out articles and consistent ROUGE implementation for evaluation. Record the model-specific preprocessing and decoding settings; this is a comparison of existing systems, not a controlled architecture-only experiment.
8. Describe the work as integration and evaluation of existing CNN/DailyMail-trained models. This does not reproduce the earlier plan of training two models from random initialization, nor learned length conditioning.
