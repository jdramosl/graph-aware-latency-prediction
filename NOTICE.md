# Third-party notices

## Derived code

The following files were adapted and extended from the code released with:

Khodabandeh, G., Ezaz, A., & Ezzati-Jivan, N. (2024). *Network Analysis of
Microservices: A Case Study on Alibaba Production Clusters.* Companion of the
15th ACM/SPEC International Conference on Performance Engineering, pp. 67-71.
Repository: https://github.com/ghazalkhb/ICPE2024_DataChallenge

- `01-Preprocessing/01-PreProcessing.py`
- `01-Preprocessing/02-Services_new.py`
- `01-Preprocessing/03-Louvain_new.py`

Academic reuse of this code was authorized by the first author, G.
Khodabandeh, by personal communication (LinkedIn message, 26 February), on
the condition that the related paper is cited. The upstream repository does
not declare a license, so these files are excluded from the MIT License in
this repository: they are shared for academic and research reuse, as
authorized, and keep the rights of their original authors. The MIT License
applies only to the code written by this repository's authors.

## Data

This repository does not redistribute data. The pipeline expects the
Alibaba Cluster Trace 2021 microservices traces (`MSCallGraph` and
`MSResource` tables), available from
https://github.com/alibaba/clusterdata/tree/master and subject to the terms
published there.
