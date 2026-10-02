# Generated artifacts

This directory holds generated output and is ignored by the root `.gitignore`
except for this guide. Create additional subdirectories as needed:

| Directory | Content |
|---|---|
| `runs/` | Per-run predictions, metrics and summaries |
| `datasets/` | Generated datasets and offline teacher requests |
| `bundles/` | Colab upload ZIPs and package builds |
| `checkpoints/` | Downloaded or exported model checkpoints |

The preserved `runs/model-forward-smoke/` and `runs/model-forward-final/` contain
untrained random-network software checks, not trained model results.

Use a new output directory for each run; do not overwrite prior research results.
Package current source from the project root with:

```bash
python scripts/package_colab.py --output artifacts/bundles/dama-model-colab.zip
```
