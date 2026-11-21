"""Fetch only the pinned BART/PEGASUS inference files into the ignored cache."""

import argparse

from hf_summarizer import MANIFEST, ROOT


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models", nargs="*", choices=list(MANIFEST), default=list(MANIFEST)
    )
    args = parser.parse_args()
    for name in args.models:
        spec = MANIFEST[name]
        print(f"Downloading {name}: {spec['id']} @ {spec['revision']}", flush=True)
        path = snapshot_download(
            spec["id"],
            revision=spec["revision"],
            cache_dir=str(ROOT / ".cache" / "huggingface" / "hub"),
            allow_patterns=["*.json", "*.txt", "*.model", spec["weight_file"]],
            # Serial first downloads avoid the Windows symlink-capability probe
            # race; the Hub then uses its ordinary non-symlink cache fallback.
            max_workers=1,
        )
        print(f"Ready: {path}", flush=True)


if __name__ == "__main__":
    main()
