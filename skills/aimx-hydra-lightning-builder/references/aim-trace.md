# Aim Trace Conventions

Aim is the evidence store for AutoResearch. `aimx` is the read-only query surface.

## Scalars

Use slash-separated names in Lightning code:

- `train/loss`
- `val/loss`
- `val/acc`
- `test/loss`

Prefer validation or test metrics for objective ranking.

Aim's Lightning logger may store the slash prefix as context, so a logged
`val/acc` can be queried as metric `acc` with context `{"subset": "val"}`.
Keep that mapping in `configs/autoresearch/default.yaml`.

## Images

Track qualitative outputs with context:

```python
aim_run.track(
    Image(fig),
    name="prediction",
    step=self.global_step,
    context={"mode": "val", "batch_idx": batch_idx},
)
```

Use config switches for image frequency and selected batches. Do not log every batch by default.

## Distributions

Track distributions only when useful:

- classifier head weights/gradients;
- activation ranges;
- feature histograms;
- residual/error distributions.

Use names such as `head/gradients/weight` and context such as `{"module": "head", "kind": "gradients"}`.

## Hyperparameters

Log these minimum sections:

- model config;
- datamodule/data config;
- trainer config;
- optimizer/scheduler config;
- plmodule config;
- task name, tags, objective, and paths.

## Aimx Queries

Use explicit repos:

```bash
aimx query params "run.hash != ''" --repo <repo> --json
aimx query metrics "metric.name == 'loss'" --repo <repo> --json
aimx trace "metric.name == 'loss'" --repo <repo> --json --tail 200
aimx query images "images.name == 'prediction'" --repo <repo> --json --head 20
```
