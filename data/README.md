# data/

`raw/` is gitignored. The MovieLens licence does not permit redistribution.

Populate it with:

```bash
python scripts/download_data.py
```

This fetches `ml-latest-small` from GroupLens, verifies its checksum, and
extracts `ratings.csv`, `movies.csv`, `tags.csv`, and `links.csv` into `raw/`.
