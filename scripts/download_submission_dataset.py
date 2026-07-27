#!/usr/bin/env python
from __future__ import annotations

from download_vidore_v3 import KNOWN_FILES, main


KNOWN_FILES.update(
    {
        "vidore/vidore_v3_industrial": {
            "corpus": [f"corpus/test-{index:05d}-of-00005.parquet" for index in range(5)],
            "queries": ["queries/test-00000-of-00001.parquet"],
            "qrels": ["qrels/test-00000-of-00001.parquet"],
        },
        "vidore/vidore_v3_pharmaceuticals": {
            "corpus": [f"corpus/test-{index:05d}-of-00002.parquet" for index in range(2)],
            "queries": ["queries/test-00000-of-00001.parquet"],
            "qrels": ["qrels/test-00000-of-00001.parquet"],
        },
        "vidore/vidore_v3_finance_fr": {
            "corpus": [f"corpus/test-{index:05d}-of-00003.parquet" for index in range(3)],
            "queries": ["queries/test-00000-of-00001.parquet"],
            "qrels": ["qrels/test-00000-of-00001.parquet"],
        },
    }
)


if __name__ == "__main__":
    main()
