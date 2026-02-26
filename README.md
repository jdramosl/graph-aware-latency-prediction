# graph-aware-latency-prediction

A graph-aware preprocessing framework that enriches cloud microservice
telemetry with dependency topology features to improve latency
prediction models.

------------------------------------------------------------------------

## Authors

Juan David Ramos López
Master’s Student
Universidad Nacional de Colombia

Arles Ernesto Rodríguez Portela, PhD
Thesis Director
Universidad Nacional de Colombia

------------------------------------------------------------------------

## Overview

This repository implements a dependency-graph--based feature engineering
pipeline for cloud-deployed microservice architectures. The framework
constructs service interaction graphs and extracts topology-aware
metrics (e.g., centrality measures and community detection) to enrich
telemetry data prior to latency prediction.

The main contribution of this work is the integration of structural
dependency information into the preprocessing stage, enabling
graph-enriched latency modeling in distributed microservice systems.

This repository accompanies the master's thesis:

**Artificial Intelligence Model for Latency Prediction in Cloud-Deployed
Microservice Architectures**



------------------------------------------------------------------------

## Acknowledgment

Parts of the preprocessing pipeline were adapted and extended from the
implementation developed in the context of the ACM/SPEC International
Conference on Performance Engineering (Companion), 2024.

Associated publication:

Khodabandeh, G., Ezaz, A., & Ezzati-Jivan, N. (2024).\
*Network Analysis of Microservices: A Case Study on Alibaba Production
Clusters.*\
Companion of the 15th ACM/SPEC International Conference on Performance
Engineering, pp. 67--71.


The following files were adapted and extended from the original
repository, with permission from the author:

-   01-Preprocessing/01-PreProcessing.py
-   01-Preprocessing/02-Services_new.py
-   01-Preprocessing/03-Louvain_new.py

These components were modified to support dependency-aware feature
enrichment and integration with the latency prediction framework
developed in this research.

All additional graph feature integration logic, data enrichment
mechanisms, and latency modeling components were independently developed
as part of this thesis.

------------------------------------------------------------------------

## Research Purpose

This repository is intended for academic and research purposes. It
provides a reproducible implementation of a graph-enriched preprocessing
pipeline for latency prediction in cloud microservice environments.
