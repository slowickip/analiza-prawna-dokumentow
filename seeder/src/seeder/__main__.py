from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from seeder.pipeline import seed_qdrant


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(
        prog="python -m seeder",
        description="Seed statutory legal units from Sejm ELI into Qdrant",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus/manifest.smoke.json"),
        help="Path to manifest JSON file",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant service URL",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=(
            "Override the collection name; by default the manifest names the "
            "collection it builds"
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Discard an already published corpus for this manifest and build again",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            Path(os.environ["ELI_CACHE_DIR"]) if "ELI_CACHE_DIR" in os.environ else None
        ),
        help="Local directory to cache raw ELI downloads",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force re-fetching from Sejm ELI and refresh the disk cache",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable ELI disk caching completely",
    )
    args = parser.parse_args()

    result = seed_qdrant(
        manifest_path=args.manifest,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        rebuild=args.rebuild,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        no_cache=args.no_cache,
    )
    if result.rebuilt:
        print(
            f"Built {result.collection} as snapshot {result.snapshot_id}: "
            f"{result.total_units} units ({result.newly_embedded} embedded)."
        )
    else:
        print(
            f"{result.collection} already publishes snapshot {result.snapshot_id} "
            f"({result.total_units} units); nothing to build."
        )


if __name__ == "__main__":
    main()
