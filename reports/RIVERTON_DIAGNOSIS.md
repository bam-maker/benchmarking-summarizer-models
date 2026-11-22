# Riverton synthetic diagnostic

This is the user's **fictional** regression article, not news and not part of the
CNN/DailyMail test benchmark. Its exact text is preserved in
[riverton.txt](../tests/fixtures/riverton.txt); all inputs, raw token IDs, decoded
outputs, settings and integrity checks are in
[final_model_verification.json](final_model_verification.json).

The article has **479 whitespace-delimited words** (`len(text.split())`), opening
“Riverton officials lifted a three-day boil-water advisory on Friday after
laboratory tests confirmed that the town’s drinking water met safety standards.”
It closes “The town plans to publish its emergency communication review within
thirty days and hold a public meeting the following week.” The fixture's trailing
file newline is excluded from the diagnostic input; all article paragraphs remain.

| Backend | Input tokens, including source specials | Source chunks / model calls | Observed main-update coverage (AI reading, not human rating) |
|---|---:|---:|---|
| LSTM | 533 | 1 / 1 | Omits the lifted advisory; emphasizes distribution, failure and precautions |
| Transformer | 533 | 1 / 1 | Mentions negative tests, but omits lifting the advisory |
| BART | 588 | 1 / 1 | Leads with lifting the advisory Friday after laboratory confirmation |
| PEGASUS | 544 | 1 / 1 | Omits lifting the advisory; describes failure and historical precautions |

All fit the unchanged 1024-token application window. No chunking, truncation,
opening removal or generative consolidation occurred. Every input character was
delivered. No arbitrary word target was requested.

**PEGASUS direct checkpoint inference has the same omission.** Its raw decoded
text describes the control-valve failure, boiling precautions and permitted
laundry/cleaning. The application displays the same three sentences, joining
sentence separators. Its displayed output exactly matches the user-reported
previous output. That earlier run's configuration is unknown; matching text
does not establish that the earlier settings were identical. Literal checkpoint
decoding and natural application settings are recorded separately. BART's direct
and application summaries both retain the principal update.

This evidence supports a **model selection/faithfulness limitation**, not a
pipeline correction. No generation changes were made for this example. Source
coverage does not imply fact coverage, and native EOS does not imply factuality.

## Integrity evidence

- Both OpenNMT originals match the official SHA-256 manifests. Every model and
  generator tensor and the saved vocabulary in the inference copies match the
  originals. Pinned model/generator loading is strict. Legacy optimizer metadata
  was read only as data for this check; it is never used for inference.
- BART and PEGASUS use their pinned tokenizer/model revisions, preserve case,
  and have no unexpected or mismatched loading keys. BART has no missing keys.
- PEGASUS's two missing positional tables exactly equal freshly computed
  deterministic sinusoidal tables and have `requires_grad=False`. The generic
  Transformers training warning does not indicate missing learned weights.
- BART's released CNN checkpoint has 50,264 embedding rows; its tokenizer also
  contains the pretraining-only `<mask>` ID 50,264. This known boundary is recorded
  explicitly, not mistaken for general vocabulary corruption. Normal Riverton
  IDs all fit; literal pretraining sentinel input is outside the tested news path.
- All modules were in evaluation mode; actual generation ran in inference mode.
  HF masks were all ones for unpadded batch size one. Every backend used CUDA FP32.
  Native generation configs remained unchanged; repeated model selection returned
  the same cached instance.
- OpenNMT still uses the existing lowercase NLTK Treebank approximation of the
  historical Stanford CoreNLP preprocessing. Earlier controlled audits document
  the differences. This remains a reproduction limitation, not a newly fixed issue.

No checkpoint, dependency, application behavior or UI was changed by these checks.
