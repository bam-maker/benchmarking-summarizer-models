# Bundled third-party assets

| Asset | Distribution | Local license |
|---|---|---|
| Reveal.js JavaScript and base CSS | `reveal.js@5.2.1`, Hakim El Hattab and Reveal.js contributors | [MIT](vendor/LICENSE-reveal.txt) |
| Source Sans 3, Latin normal 400 and 700 | `@fontsource/source-sans-3@5.2.5`, Adobe Source Sans family | [SIL Open Font License 1.1](vendor/LICENSE-source.txt) |
| IBM Plex Mono, Latin normal 500 | `@fontsource/ibm-plex-mono@5.2.5`, IBM Plex family | [SIL Open Font License 1.1](vendor/LICENSE-plex.txt) |
| DejaVu Sans, fallback for arrows, subscripts and mathematical symbols | Unmodified font from Matplotlib v3.10.0 | [DejaVu / Bitstream / Arev notices](vendor/LICENSE-dejavu.txt) |

The Reveal.js and Fontsource files were extracted from the published npm tarballs after checking the registry’s SHA-512 integrity values. DejaVu Sans and its license came from the tagged Matplotlib source tree. The deck loads every runtime asset from the local `vendor` folder. [Reveal.js](https://github.com/hakimel/reveal.js) · [Source Sans](https://github.com/adobe-fonts/source-sans) · [IBM Plex](https://github.com/IBM/plex) · [DejaVu distribution](https://github.com/matplotlib/matplotlib/tree/v3.10.0/lib/matplotlib/mpl-data/fonts/ttf)
