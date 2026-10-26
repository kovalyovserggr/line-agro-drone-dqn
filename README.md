# line-agro-drone-dqn

Simulation, code and experimental data for the article on **DQN-based optimization of efficiency and reliability of an automated agrodrone production line** (Article 3 of the dissertation series *"Optimization of Efficiency and Reliability of Automated Production Systems Using Artificial Intelligence Methods"*, Central Ukrainian National Technical University).

> Working branch and default branch: `article3-dqn`.
> Every reported experiment is tied to a git tag `a3-*` and a row in `experiments/article3/runs_index.md`.

## What is here

| Path | Content |
|---|---|
| `experiments/article3/` | **Everything for Article 3**: training, evaluation and baseline scripts, figures/tables generation, run logs |
| `experiments/article3/runs_index.md` | Index of all runs: run_id, date, tag, full command, environment and agent seeds, parameters, files, status |
| `experiments/article3/RESULTS_LOG.md` | Chronological log of results and decisions |
| `experiments/article3/raw/` | Raw metrics (CSV) and agent folders of the runs |
| `src/`, `tests/` | Simulation model of the production line (`AgroDroneLine`) and its tests |
| `experiments/run_parametric_study.py`, `experiments/runs_index.md`, `experiments/RESULTS_LOG.md` | **Inherited from Article 1** (see below) |

## Inherited Article 1 files

This repository was created from the local clone of the Article 1 repository (`kovalyovserggr/line-agro-drone`), so its history is shared. Files outside `experiments/article3/` that belong to Article 1 (`experiments/run_parametric_study.py`, `experiments/runs_index.md`, `experiments/RESULTS_LOG.md`) describe the T_pm parametric study of the simulation and are **not** part of Article 3 results. The Article 1 data are published separately in the Article 1 repository (tag `paper1-data-v1`).

## Reproducing results

1. Python environment with the dependencies of the simulation and of the agent (PyTorch).
2. Check out the tag of the run you need, e.g. `git checkout a3-...` (list: `git tag -l "a3-*"`).
3. Take the **full command** from the row of `experiments/article3/runs_index.md` and run it from the repository root.

The simulation is deterministic for a given seed. Training uses separate seeds for the environment and for the agent; evaluation uses a fixed set of 20 test seeds (100000–100019) that are not used in training. Seeds of every run are listed in `runs_index.md`.

## Event logs

Large event logs are not stored in git. They are attached to the repository **Release** as a zip archive with a SHA256 checksum (also recorded in `runs_index.md`). Verify after download:

```powershell
Get-FileHash <archive>.zip -Algorithm SHA256
```

## Citation

See `CITATION.cff` (GitHub shows a "Cite this repository" button). After the article is accepted, the final state will be tagged `paper3-v1` and archived with a DOI.

## License

MIT, see `LICENSE`.
