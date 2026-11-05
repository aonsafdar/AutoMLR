# Behavioural Signatures of a Language-Model Agent in Autonomous Vision-Architecture Research

**Aon Safdar** and **Mohamed Saadeldin**
University College Dublin, Ireland

Accepted at the **NeurIPS 2026 Workshop for Autonomous ML Research (AutoMLR)**.
An extended preprint with full appendices is available on arXiv:
[arXiv:2608.01995](https://arxiv.org/abs/2608.01995).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What this repository is

Code and research artefacts accompanying the paper. A single general-purpose
language-model agent was given the full role of researcher on one open-ended
problem: designing a vision transformer that mixes information purely through
channel attention, with no spatial self-attention at any stage. Over three
phases and 111 sequential hypotheses it took the design from a CIFAR-10 baseline
to 79.83% top-1 on ImageNet-1K at 3.7 GFLOPs.

The paper's object of study is the agent's behaviour, not the architecture.
This repository therefore contains everything needed to (a) inspect the model,
training driver, and per-run configurations produced during the study, and
(b) audit the complete per-hypothesis research trace that the paper analyses:
the human-authored instruction documents, the agent's append-only research log,
the parsed hypothesis table, the figure-generation scripts, the champion
configurations, and the Slurm templates used for the control hand-off.

Site-specific identifiers (absolute paths, cluster names) have been replaced with
`/path/to/...` placeholders; nothing else has been altered.

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
appendix.pdf               # supplementary material: implementation, full hypothesis log, extra tables/figures
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

## Citation

```bibtex
@inproceedings{safdar2026behavioural,
  title     = {Behavioural Signatures of a Language-Model Agent in Autonomous Vision-Architecture Research},
  author    = {Safdar, Aon and Saadeldin, Mohamed},
  booktitle = {NeurIPS 2026 Workshop for Autonomous ML Research (AutoMLR)},
  year      = {2026}
}
```

## Acknowledgements

This work was supported by Taighde Éireann – Research Ireland under Grant 18/CRT/6183.

## License

MIT. See `LICENSE`.
