# Server Runbook: AdaColRAG Paper Finalization

This runbook assumes that the four-dataset main experiments and four final-candidate experiments already exist and have passed validation. The remaining workflow performs only statistical postprocessing, single-process timing, manuscript-table generation, and packaging.

## 1. Save local edits and pull the final branch

```bash
cd ~/GMY/AdaColRAG

BACKUP_DIR="$HOME/GMY/adacolrag_before_paper_final_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
git status --short > "$BACKUP_DIR/git_status.txt"
git diff > "$BACKUP_DIR/tracked_changes.patch" || true
cp -a paper/draft.md "$BACKUP_DIR/draft.md" 2>/dev/null || true

git restore --source=HEAD --staged --worktree .
git fetch origin
git switch agent/adacolrag-wsdm-framework
git pull --ff-only origin agent/adacolrag-wsdm-framework
git log -1 --oneline
```

Do not restore older copies of `paper/draft.md`, `configs/adacolrag.yaml`, or finalization scripts over the pulled versions.

## 2. Refresh the core environment

```bash
cd ~/GMY/AdaColRAG
conda run --no-capture-output -n adacolrag-core python -m pip install -e .
```

No ColPali or VisRAG model execution is required for finalization.

## 3. Verify code and the frozen final configuration

```bash
cd ~/GMY/AdaColRAG

conda run --no-capture-output -n adacolrag-core pytest -q
conda run --no-capture-output -n adacolrag-core python -m compileall -q src scripts tests

bash -n scripts/run_paper_finalization.sh
bash -n scripts/package_paper_finalization_review.sh
chmod +x scripts/run_paper_finalization.sh scripts/package_paper_finalization_review.sh
```

Verify the final configuration:

```bash
conda run --no-capture-output -n adacolrag-core python - <<'PY'
from adacolrag.config import load_config

config = load_config("configs/adacolrag.yaml")
print(config)
assert config["mode"] == "fixed_mmr"
assert config["fixed_tokens"] == 112
assert config["selector"]["redundancy_weight"] == 0.25
assert config["selector"]["layout_weight"] == 0.0
assert config["selector"]["prefilter_factor"] == 4.0
assert config["fusion"]["dense_weight"] == 0.25
assert config["fallback"]["enabled"] is False
print("final configuration: PASS")
PY
```

## 4. Verify existing effectiveness evidence

```bash
cd ~/GMY/AdaColRAG

python - <<'PY'
import json
from pathlib import Path

datasets = (
    "vidore_v3_finance_en",
    "vidore_v3_industrial",
    "vidore_v3_pharmaceuticals",
    "vidore_v3_finance_fr",
)
for slug in datasets:
    for root_name, expected in (("results/submission", 13), ("results/final_targeted", 4)):
        root = Path(root_name) / slug
        validation = root / "validation.json"
        payload = json.loads(validation.read_text(encoding="utf-8"))
        runs = list((root / "runs").glob("*.json"))
        print(slug, root_name, "valid=", payload.get("valid"), "runs=", len(runs))
        assert payload.get("valid") is True
        assert not payload.get("errors")
        assert len(runs) == expected
print("stored effectiveness evidence: PASS")
PY
```

## 5. Run final statistical and timing workflow

Use `tmux`:

```bash
tmux new -s adacolrag-paper-final
```

Inside the session:

```bash
cd ~/GMY/AdaColRAG

PROJECT_ROOT="$HOME/GMY/AdaColRAG" \
BOOTSTRAP_SAMPLES=10000 \
STEADY_REPEATS=7 \
STEADY_WARMUPS=1 \
STEADY_METHODS="visrag_dense,colpali_full,visrag_top50_full,adacolrag" \
CPU_AFFINITY="0-7" \
USE_TASKSET=1 \
OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
bash scripts/run_paper_finalization.sh \
2>&1 | tee logs/paper_final_full_workflow.log
```

This command does not download datasets, load model checkpoints, regenerate embeddings, or rerun the 68 effectiveness experiments.

Detach with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t adacolrag-paper-final
```

## 6. Monitor progress

```bash
tail -f logs/paper_final/vidore_v3_finance_en/01_final_significance.log
tail -f logs/paper_final/vidore_v3_finance_en/02_steady_state_timing.log
tail -f logs/paper_final/vidore_v3_industrial/02_steady_state_timing.log
tail -f logs/paper_final/vidore_v3_pharmaceuticals/02_steady_state_timing.log
tail -f logs/paper_final/vidore_v3_finance_fr/02_steady_state_timing.log
```

## 7. Verify final outputs

```bash
cd ~/GMY/AdaColRAG

python - <<'PY'
import json
from pathlib import Path

datasets = (
    "vidore_v3_finance_en",
    "vidore_v3_industrial",
    "vidore_v3_pharmaceuticals",
    "vidore_v3_finance_fr",
)
for slug in datasets:
    root = Path("results/paper_final") / slug
    significance = root / "significance" / "final_vs_baselines.json"
    timing = root / "steady_state_timing.json"
    assert significance.is_file() and significance.stat().st_size > 0
    assert timing.is_file() and timing.stat().st_size > 0
    payload = json.loads(timing.read_text(encoding="utf-8"))
    assert payload["timing_mode"] == "single_process_steady_state"
    assert set(payload["experiments"]) == {
        "visrag_dense", "colpali_full", "visrag_top50_full", "adacolrag"
    }
    for result in payload["experiments"].values():
        assert result["repeats"] == 7
        assert result["warmups"] == 1
    print(slug, "paper-final evidence: PASS")

for name in (
    "final_main_results.md",
    "final_significance.md",
    "final_efficiency.md",
    "final_steady_timing.md",
    "final_selection.md",
    "final_paper_results.json",
):
    path = Path("paper/generated") / name
    assert path.is_file() and path.stat().st_size > 0, path
print("generated manuscript tables: PASS")
PY
```

Check the manuscript:

```bash
grep -RInE '^(<<<<<<<|=======|>>>>>>>)' paper/draft.md \
  || echo "paper/draft.md has no conflict markers"

grep -n 'BEGIN AUTO:FINAL_' paper/draft.md
```

## 8. Review key tables

```bash
cat paper/generated/final_selection.md
cat paper/generated/final_main_results.md
cat paper/generated/final_significance.md
cat paper/generated/final_efficiency.md
cat paper/generated/final_steady_timing.md
```

## 9. Package the final review bundle

Tracked code should be clean before packaging:

```bash
cd ~/GMY/AdaColRAG
git status -sb
```

Then run:

```bash
OUTPUT="AdaColRAG_paper_finalization_review_bundle.tar.gz" \
bash scripts/package_paper_finalization_review.sh
```

The output archive contains validated result JSON files, final statistics, steady-state timing, manifests, configurations, scripts, generated tables, and the manuscript. It excludes model weights, parquet files, page images, and embedding arrays.

## 10. Important final paths

```text
results/submission/<dataset>/
results/final_targeted/<dataset>/
results/paper_final/<dataset>/significance/
results/paper_final/<dataset>/steady_state_timing.json
paper/generated/final_*.md
paper/generated/final_paper_results.json
paper/draft.md
AdaColRAG_paper_finalization_review_bundle.tar.gz
```
