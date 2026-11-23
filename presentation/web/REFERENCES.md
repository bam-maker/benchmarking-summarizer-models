# Presentation references and evidence

The active browser deck uses the final training scores supplied by the user, who confirmed the same 250 articles and full-article evaluation. Main charts and the final result table use these values. Earlier measured repository benchmarks remain in labeled backup slides. [Presentation scores and provenance](reported_scores.json)

## Primary references

The dedicated reference slides use these numbers. Full publication details and checkpoint URLs are in the existing presentation reference file. [Publication details and original evidence map](../REFERENCES.md)

| Reference | Source and use |
|---|---|
| 1 | CNN/DailyMail 3.0.0 dataset card and Hermann et al. (2015). Dataset split counts, article/highlight pairing and the saved test-row example. |
| 2 | OpenNMT-py 0.2.1 summarization recipe and released CNN/DailyMail checkpoints. Architecture, historical PTB/lowercase preparation, search and copy mechanism. |
| 3 | Hochreiter and Schmidhuber (1997), Long Short-Term Memory; Bahdanau et al. (2015), attention. |
| 4 | Vaswani et al. (2017), Attention Is All You Need. Self-attention, masking and source cross-attention. |
| 5 | Lin (2004), ROUGE; Google Research rouge-score implementation. Four separate F1 measures with Porter stemming. |
| 6 | Lewis et al. (2020), BART; facebook/bart-large-cnn pinned release. Denoising pretraining and CNN/DailyMail fine-tuning. |
| 7 | Zhang et al. (2020), PEGASUS; released Mixed & Stochastic CNN/DailyMail model. Gap-sentence pretraining, C4 and HugeNews. |

## Repository evidence

- The earlier repository benchmark configuration records checkpoint identities, hashes, generation settings, dependency inventory and code hashes. Its two tracks, original references and predictions supply the historical backup tables. The supplied training scores are recorded separately in the presentation source file. [Earlier benchmark250](../../reports/benchmark250/README.md) · [Presentation scores](reported_scores.json)
- The small worked preprocessing example uses the earlier saved Riverton fixture. Its lowercase words and BART ID prefix are measured. The word-vocabulary IDs marked Illustrative explain the mapping without claiming to be saved OpenNMT IDs. [Saved tokenizer trace](../../reports/final_model_verification.json)
- The app normalizes typographic quotes, uses NLTK Treebank and loads the existing checkpoint vocabulary. The OpenNMT inputter builds an article-specific copy dictionary and torchtext input fields. Its batch size is one. Hugging Face adapters retain their own tokenizers and use input tensors with attention masks. [OpenNMT adapter](../../summarizer.py) · [HF adapter](../../hf_summarizer.py)
- Evaluation preserves the original highlights, checks exact text hashes and retains the original seed-42 order. It uses the existing Punkt sentence-span function for ROUGE-Lsum segmentation. [Frozen protocol](../../tests/final_benchmark.py) · [250-article extension](../../tests/benchmark250.py)
- The length-control and consolidation observations come from saved development comparisons. The current natural policy and guarded 1,024-token OpenNMT work window are recorded in the project status and production implementation. These experiments describe engineering choices without establishing a benchmark improvement. [Regression evidence](../../reports/quality_regression.json) · [Natural-policy checks](../../reports/natural_policy_verification.json) · [Full-article implementation](../../article_summarizer.py) · [Historical design notes](../../FULL_ARTICLE.md)
- Saved Riverton examples use the latest fictional 315-word body and reproduced outputs. The identical original display remains unresolved. [Latest diagnosis](../../reports/LAPTOP_MODEL_DIAGNOSIS.md)
- Architecture diagrams, configurations, the live app and saved diagnostic outputs describe the released app checkpoints. The user supplied aggregate scores for the final training evaluation. Checkpoint files, training logs and per-article predictions for that evaluation were not supplied. [App checkpoint manifest](../../OpenNMT_Verified_Setup/checkpoint_manifest.json) · [App HF revisions](../../model_manifest.json) · [Presentation scores](reported_scores.json)
