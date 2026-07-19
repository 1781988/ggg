# Results policy

`results/smoke_reference.json` is a deterministic implementation check. It is not a benchmark result and must not be cited as empirical evidence.

Real experiment outputs belong under:

```text
results/real/<dataset>/<experiment>.json
```

Each file must be produced by `adacolrag evaluate` and retain:

- config hash;
- dataset checksum;
- model artifact paths/manifests;
- query-level rankings, confidence, fallback state, and latency;
- aggregate quality and efficiency metrics.

Do not edit generated metrics manually. Failed and negative results should be preserved. Paper tables should be generated from these JSON files after the experiment matrix is frozen.
