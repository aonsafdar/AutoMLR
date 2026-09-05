# Autonomous Architecture Research with a Language-Model Agent

Code and research artifacts accompanying the anonymous double-blind workshop
submission:

> **Behavioural Signatures of a Language-Model Agent in Autonomous Vision-Architecture Research**

This repository is the browsable mirror of the submission's supplementary
material. It contains everything needed to (a) inspect the model, training
driver, and per-run configurations produced during the study, and (b) audit the
full per-hypothesis research trace that the paper analyses.

> **Anonymity notice.** This repository is provided for double-blind review and
> is intentionally free of author, institution, and cluster identifiers.
> Absolute paths have been replaced with `/path/to/...` placeholders. A
> permanent, de-anonymized public release will replace this snapshot upon
> acceptance.

## Repository layout

```
main.py                    # training / evaluation driver (DDP-ready)
config.py                  # YACS config system + defaults
optimizer.py, lr_scheduler.py, logger.py, utils.py
models/                    # backbone + baselines (channel-attention ViT, CvT, ViT, CaiT, Swin)
configs/                   # per-run YAML configs (champions + baselines)
tools/                     # FLOPs/throughput profiling + attention visualisation
figures/                   # scripts that regenerate the paper's figures from the log
log/                       # research_log.md (full trace) + hypotheses.csv
instructions/              # research brief, program, agent instructions, tooling
slurm/                     # scheduler job templates (1-/2-GPU, watcher, submit-and-wait)
outputs/                   # architecture_evolution.md (narrative of the champion lineage)
appendix.pdf               # extended appendix: implementation, full hypothesis log, extra tables/figures
requirements.txt           # Python dependencies
setup_env.sh               # conda environment bootstrap
```

## Auditing the research trace

The heart of the study is `log/research_log.md`, the agent's append-only working
memory (one entry per hypothesis), together with `log/hypotheses.csv` for a
tabular view. `figures/build_figures.py` regenerates the paper's trajectory and
rolling-success-rate figures directly from these logs, so every quantitative
claim can be reproduced from source.

## Quickstart

```bash
# 1. Environment
bash setup_env.sh          # creates a conda env with all dependencies
conda activate automlr

# 2. Point the config's DATA_PATH at your dataset, then train
python -m torch.distributed.run --nproc_per_node=1 main.py \
    --cfg configs/imagenet_100ep_baseline.yaml

# Weights & Biases logging is optional; disable with WANDB=0.
```

Configs use a YACS-style hierarchy; override any field on the command line via
`--opts KEY value ...`. The `DATA_PATH` fields are placeholders and must be set
to your local CIFAR / ImageNet-1K locations.

## Scope

The channel-attention backbone and its baselines are research artifacts of an
autonomous, agent-driven search; they are provided for reproducibility and
inspection of the study, not as a tuned, production model release.
