# Aim Compatibility and Real-Read Verification

**Date**: 2026-10-29  
**Branch**: `008-aimx-retro`  
**Native Aim Version**: `3.29.1`  
**Aimx Version**: `0.3.1`  
**Environment**: Python `3.12.7` on Darwin arm64  

---

## 1. Verified Native Aim Commands

Tested against real repository data at `data/.aim`:

```bash
uv run aim runs --repo data/.aim ls
# Output: 15 runs enumerated cleanly
```

---

## 2. Verified Aimx Read Queries

The retro skill evidence guide prescribes the following query commands. All executed successfully with zero repository mutations:

### Parameter Query
```bash
uv run aimx query params "run.hash == '0444bd8c227f4678bdb54da2'" --repo data/.aim --json
```
- Status: Exit `0`
- Runs returned: 1
- Extracted parameters: `hparam.loss_fn`, `hparam.lr` (0.0005), `hparam.optimizer` (AdamW), `model` (ResNet).

### Metric Query
```bash
uv run aimx query metrics --repo data/.aim --json
```
- Status: Exit `0`
- 15 runs, 168 metrics inspected across subsets (`val`, `train`).

### Trace Query
```bash
uv run aimx trace "(run.hash == 'eca37394eeb84f48a5d2d736') and metric.name == 'loss'" --repo data/.aim --json --tail 5
```
- Status: Exit `0`
- Returns bounded 5-step tail values for train and val loss.

---

## 3. Data Integrity & Read-Only Invariance

- Source directory `data/.aim` remained untouched before and after read operations.
- Zero sidecar or mutation files were created in `data/`.
