# graph-aware-latency-prediction

A graph-aware preprocessing framework that enriches cloud microservice
telemetry with dependency topology features to improve latency
prediction models.

------------------------------------------------------------------------

## Authors

Juan David Ramos López, Master's Student, Universidad Nacional de Colombia

Arles Ernesto Rodríguez Portela, PhD, Thesis Director, Departamento de
Matemáticas, Facultad de Ciencias, Universidad Nacional de Colombia

## Project

Developed within the research project *Modelado Computacional de Sistemas
Complejos: Analizando el Impacto del Intercambio de Información en la Toma
de Decisiones* (Hermes code 64022), Departamento de Matemáticas, Facultad de
Ciencias, Universidad Nacional de Colombia (Bogotá). This repository is the
open-source software deliverable of the project's microservices case study.

## Overview

This repository implements a dependency-graph-based feature engineering
pipeline for cloud-deployed microservice architectures. The framework
constructs per-request service interaction graphs and extracts
topology-aware metrics (centrality measures, community detection and
global graph metrics) to enrich telemetry data prior to latency
prediction. Five supervised models (Linear Regression, Decision Tree,
Random Forest, XGBoost and an LSTM) are compared with and without the
structural features, and SHAP is used to interpret their contribution.

It accompanies the master's thesis and the article:

> Ramos-López, J., & Rodríguez, A. *Artificial Intelligence Model for
> Latency Prediction in Cloud-Deployed Microservice Architectures.*
> Submitted to Future Generation Computer Systems (2026).

## Repository layout

| Folder | Content |
|---|---|
| `01-Preprocessing/` | Merging of call-graph and resource tables, per-service graphs, Louvain communities and centrality metrics, final enriched dataset |
| `02-Training/` | Training and evaluation of the tree-based/linear models and of the LSTM (`mae_train_models.py`, `train_lstm.py`) |
| `03-Interpretability/` | SHAP analysis (`shap_analysis_c.py`, `lstm_shap_lstm.ipynb`) |

## Data

The data are **not** included. The pipeline uses the Alibaba Cluster Trace
2021 microservices traces (`MSCallGraph_*.csv` and `MSResource_*.csv`):
<https://github.com/alibaba/clusterdata/tree/master>.

## Pipeline, in order

1. `01-Preprocessing/00-FilterMatching.py`: joins call-graph and resource
   files in parallel; writes `filtered_data_um.csv`.
2. `01-Preprocessing/01-PreProcessing.py`: keeps and renames the columns
   (`source`, `target`, `timestamp`, `service`, `traceid`, `rt`); writes
   `filtered_data_um_sorted.csv`.
3. `01-Preprocessing/02-Services_new.py`: splits the data into one CSV per
   service (`services_new/<service>_output.csv`).
4. `01-Preprocessing/03-Louvain_new.py`: builds each graph, takes its largest
   connected component and computes betweenness, closeness, degree,
   eigenvector, PageRank, eccentricity and Louvain community (seed 650), plus
   global metrics (nodes, edges, density, clustering, average degree,
   diameter, radius); writes `louvain_new/`.
5. `01-Preprocessing/04-Parallel.py`: attaches the graph features to every
   record in parallel; writes
   `filtered_data_um_with_extra_columns_parallel.csv`.
6. `02-Training/mae_train_models.py` and `02-Training/train_lstm.py`: train
   and evaluate the models under the baseline (resource metrics) and the
   enriched (resource + structural metrics) scenarios; split 80/20 with
   `random_state=42`.
7. `03-Interpretability/`: SHAP values for the trained models.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The authors' notebook reports Python 3.11.5.

## Known limitations

- Input and output paths are hard-coded at the top of the scripts (absolute
  Windows and Linux paths from the authors' machines, e.g. `D:\preprocess_sem_5\...`
  or `/home/juan/...`). Edit them before running; moving them to a
  configuration file or command-line arguments is pending.
- Package versions are not pinned yet.

## Acknowledgment

Parts of the preprocessing pipeline are derived from the code released with:

Khodabandeh, G., Ezaz, A., & Ezzati-Jivan, N. (2024). *Network Analysis of
Microservices: A Case Study on Alibaba Production Clusters.* Companion of the
15th ACM/SPEC International Conference on Performance Engineering, pp. 67-71.
Code: <https://github.com/ghazalkhb/ICPE2024_DataChallenge>

The following files were adapted and extended from that repository. Academic
reuse was authorized by the first author (G. Khodabandeh, personal
communication via LinkedIn, 26 February) on the condition that the related
paper is cited:

- `01-Preprocessing/01-PreProcessing.py` (from `PreProcessing.py`)
- `01-Preprocessing/02-Services_new.py` (from `Services.py`)
- `01-Preprocessing/03-Louvain_new.py` (from `Louvain.py`)

The upstream repository does not declare a license, so these three files are
**not** covered by the MIT License of this repository: they are shared for
academic and research reuse, as authorized, and keep the rights of their
original authors.

All additional graph feature integration logic, data enrichment mechanisms
and latency modeling components were developed as part of this thesis. See
`NOTICE.md` for details.

## License

Released under the MIT License (see `LICENSE`), except for the three files
derived from third-party code listed in the Acknowledgment.
