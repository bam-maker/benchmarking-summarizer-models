"""Real cached-model checks for complete coverage and the new application path."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import article_summarizer as full
import summarizer as engine


def long_article():
    projects = (
        "electric buses",
        "public libraries",
        "flood barriers",
        "solar panels",
        "health clinics",
        "cycling routes",
        "school buildings",
        "water pipes",
        "recycling facilities",
        "rail stations",
        "city parks",
        "emergency shelters",
    )
    return "\n\n".join(
        f"The council in district {i + 1} has approved a plan to improve {project} over the next three years. "
        f"The first phase will begin in September and will cost {20 + i * 5} million pounds, officials said on Tuesday. "
        "The work will be funded by the local council and a national transport and infrastructure grant. "
        "A public consultation found that residents supported the project but asked officials to limit construction noise. "
        "Contractors must publish monthly progress reports and meet independent safety standards. "
        "Existing public services will remain available during construction. "
        "The council will review the budget after the first year and publish the results."
        for i, project in enumerate(projects)
    )


def main():
    config = full.load_config()
    article = long_article()
    rows = []
    identities = {}
    for name in engine.MODEL_CHOICES:
        short = full.summarize_article(
            "The council approved new buses.", name, config=config
        )
        assert short.returned_input and short.coverage_complete
        for length in full.LENGTH_CHOICES:
            result = full.summarize_article(article, name, length, config=config)
            assert result.summary and result.coverage_complete
            assert result.covered_characters == len(article)
            assert result.source_chunks > 1
            assert result.model_calls >= result.source_chunks
            assert 0 <= result.consolidation_passes <= config.max_consolidation_passes
            if result.consolidation_passes == 0:
                assert result.actual_words <= result.requested_words
            assert result.requested_words == full.requested_words(
                len(article.split()), length, config
            )
            assert all(
                s.output_words < s.input_words and s.output_tokens < s.input_tokens
                for s in result.stages
            )
            backend = engine._models[name]
            identities.setdefault(name, id(backend))
            assert identities[name] == id(backend)
            row = asdict(result)
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in {"summary", "chunk_spans", "stages"}
                    }
                ),
                flush=True,
            )

        oversized = " ".join(article.replace(".", "").split())
        chunks = full.ChunkPlanner(backend).plan(oversized, config.max_source_chunks)
        assert len(chunks) > 1 and "".join(c.text for c in chunks) == oversized
        assert all(not c.prepared.truncated for c in chunks)
        oversized_result = full.summarize_article(
            oversized, name, "Medium", config=config
        )
        assert oversized_result.summary and oversized_result.coverage_complete
        rows.append(asdict(oversized_result))
        assert id(engine._models[name]) == identities[name]

    try:
        full.summarize_article(
            article, "LSTM", config=replace(config, max_model_calls=1)
        )
    except full.ProcessingLimitError:
        pass
    else:
        raise AssertionError(
            "A workload-limited request must not return a partial result"
        )
    path = ROOT / "test-results" / "full_article.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "kind": "application integration checks; not benchmark scores",
                "cases": rows,
                "complete_source_coverage": True,
                "model_identity_reused": True,
                "workload_limit_rejected": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"PASS: 12 real model/length long-article cases, 4 oversized sentences, 4 explicit concise-input cases; {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
