# References and evidence

Prepared for Biruk Anley and Daniela Ortega. The redesign uses the repository
evidence present at commit `e016c88`, with byte-level source hashes recorded in
[the evidence manifest](assets/evidence_manifest.json). Dataset counts, the pinned
BART card, the PEGASUS release card and the ROUGE implementation were checked
against primary sources on November 22, 2026.

Slide citation numbers map to the dataset card [1], OpenNMT release [2], LSTM
paper [3], Transformer paper [4], ROUGE [5], BART [6] and PEGASUS [7].

The CNN/DailyMail card's configuration 3.0.0 lists 287,113 training, 13,368
validation and 11,490 test rows. The actual saved example on slide 3 belongs to
the fixed test sample. Its ID, exact excerpts and original hashes are preserved
in [dataset_example.json](assets/dataset_example.json). The display uses excerpts.
The complete original article and reference remain in the local saved corpus.
[Dataset configuration source](https://huggingface.co/datasets/abisee/cnn_dailymail/blob/main/README.md)

## Architecture and pretraining

- Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory.**
  *Neural Computation*, 9(8), 1735–1780.
  [Original paper/DOI](https://doi.org/10.1162/neco.1997.9.8.1735).
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need.**
  *Advances in Neural Information Processing Systems 30*.
  [Proceedings](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). **Neural Machine Translation by
  Jointly Learning to Align and Translate.** *ICLR* (preprint 2014).
  [Authors' paper](https://arxiv.org/abs/1409.0473).
- See, A., Liu, P. J., & Manning, C. D. (2017). **Get To The Point: Summarization
  with Pointer-Generator Networks.** *ACL*, 1073–1083.
  [Paper](https://aclanthology.org/P17-1099/). Related copy-mixture literature;
  our checkpoints are the actual OpenNMT releases, not a claim to reproduce the
  paper's complete coverage model. Neither has learned coverage attention.
- Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O.,
  Stoyanov, V., & Zettlemoyer, L. (2020). **BART: Denoising Sequence-to-Sequence
  Pre-training for Natural Language Generation, Translation, and Comprehension.**
  *ACL*, 7871–7880. [Paper](https://aclanthology.org/2020.acl-main.703/).
- Zhang, J., Zhao, Y., Saleh, M., & Liu, P. J. (2020). **PEGASUS: Pre-training
  with Extracted Gap-sentences for Abstractive Summarization.**
  *ICML*, PMLR 119, 11328–11339.
  [Paper](https://proceedings.mlr.press/v119/zhang20ae.html).

## Dataset and evaluation

- Hermann, K. M., Kočiský, T., Grefenstette, E., Espeholt, L., Kay, W.,
  Suleyman, M., & Blunsom, P. (2015). **Teaching Machines to Read and Comprehend.**
  *Advances in Neural Information Processing Systems 28*.
  [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2015/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html).
  This is the original reading-comprehension dataset source.
- Nallapati, R., Zhou, B., dos Santos, C., Gülçehre, Ç., & Xiang, B. (2016).
  **Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond.**
  *CoNLL*, 280–290. [Paper](https://aclanthology.org/K16-1028/).
- [CNN/DailyMail dataset card](https://huggingface.co/datasets/abisee/cnn_dailymail)
  and [original summarization data preparation](https://github.com/abisee/cnn-dailymail).
  Our evaluation uses the saved 3.0.0 test sample; historical checkpoint training
  split and preprocessing identity across all released models is not assumed.
- Lin, C.-Y. (2004). **ROUGE: A Package for Automatic Evaluation of Summaries.**
  *Text Summarization Branches Out*, 74–81.
  [Paper](https://aclanthology.org/W04-1013/).
- Google Research. [Python ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge).
  Documents `rougeL` versus sentence-aware union-LCS `rougeLsum`. Saved project
  metrics are F1 ×100 with Porter stemming; no confidence intervals or human
  factuality ratings are supplied.

## Released checkpoint sources

- [OpenNMT summarization recipe](https://opennmt.net/OpenNMT-py/examples/summary/Summarization.html).
  Repository compatibility checkout: OpenNMT-py `0.2.1`, commit
  `6db7ec1b3afae28c257a0416dd947022c3b12bc0`. Its pinned local recipe establishes
  the search settings; the current website may have different defaults.
- LSTM official checkpoint:
  [`ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt`](https://s3.amazonaws.com/opennmt-models/Summary/ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt).
  Original SHA-256: `7e0092b066014af8b02c1221853ad48f0ee0db2182787baeb40abd9df2331d2e`.
- Transformer official checkpoint:
  [`sum_transformer_model_acc_57.25_ppl_9.22_e16.pt`](https://s3.amazonaws.com/opennmt-models/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt).
  Original SHA-256: `683ef35e72d47cb33043e5db243b78a1386dd8a57e079c84b7642fba126c6605`.
- [BART pinned checkpoint card](https://huggingface.co/facebook/bart-large-cnn/blob/37f520fa929c961707657b28798b30c003dd100b/README.md):
  `facebook/bart-large-cnn`, revision `37f520fa929c961707657b28798b30c003dd100b`.
  Broad denoising pretraining, then CNN/DailyMail fine-tuning.
- [PEGASUS checkpoint card](https://huggingface.co/google/pegasus-cnn_dailymail)
  and [authors' release code](https://github.com/google-research/pegasus):
  `google/pegasus-cnn_dailymail`, revision `40d588fdab0cc077b80d950b300bf66ad3c75b92`.
  The released model card describes updated **Mixed & Stochastic** pretraining
  on C4 and HugeNews, followed by CNN/DailyMail fine-tuning. This should not be
  confused with a single-corpus experimental model from the original paper.

## Repository evidence mapped to the presentation

| Content | Source |
|---|---|
| Current policy, offline Ubuntu status, no length selector | [PROJECT_STATUS.md](../PROJECT_STATUS.md), [README.md](../README.md), [app.py](../app.py) |
| Actual OpenNMT architecture, copy mechanism, parameter counts and limits | [MODEL_CAPABILITIES.md](../MODEL_CAPABILITIES.md), [laptop diagnosis](../reports/LAPTOP_MODEL_DIAGNOSIS.md), [summarizer.py](../summarizer.py) |
| Saved word-token example from the earlier fictional fixture | [model verification](../reports/final_model_verification.json), `models.LSTM.chunks[0].prepared_tokens` |
| Length-control experiments and subsequent policy | [FULL_ARTICLE.md](../FULL_ARTICLE.md), [quality regression](../reports/quality_regression.json), [current policy checks](../reports/natural_policy_verification.json) |
| Full and shared-prefix ROUGE values | [saved JSON](../reports/pilot50/results.json), [benchmark report](../BENCHMARK_RESULTS.md) |
| Latest exact fictional article and saved outputs | [laptop diagnosis](../reports/LAPTOP_MODEL_DIAGNOSIS.md) |
| HF checkpoint identity and adapters | [model_manifest.json](../model_manifest.json), [hf_summarizer.py](../hf_summarizer.py) |
| Original checkpoint hashes | [OpenNMT manifest](../OpenNMT_Verified_Setup/checkpoint_manifest.json) |
| Full processing and bounded consolidation | [article_summarizer.py](../article_summarizer.py), [processing_config.json](../processing_config.json) |

`MODEL_CAPABILITIES.md` also preserves older adaptive-length experiments. The
current application and this presentation use `natural-checkpoint-v1` as recorded
in `PROJECT_STATUS.md`; historical length selectors or budgets were not imported
into the current interface schematic.
