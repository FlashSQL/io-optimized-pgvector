# IO-optimized-pgvector

## Overview
This repository provides an optimized implementation of pgvector with SSD-aware enhancements for high-performance vector search.
The optimizations focus on improving I/O efficiency, leveraging SSD parallelism, query reordering, and locality-preserving indexing.

## Branches
This repository consists of five branches:

- **main**: Includes README file.
- **pg_orig**: Vanilla (original) PostgreSQL + pgvector.
- **pg_iou**: Asynchronous I/O using IOUring.
- **pg_async_iou**: Asynchronous I/O + overlapping execution.
- **pg_colocation**: Partitioning + locality-aware insertion.

## Prerequisites
To install the required dependencies, run:
```sh
apt install zlib1g-dev flex bison libreadline-dev gdb rsync liburing-dev
```

## PostgreSQL & pgvector
This project builds upon the following open-source projects:
- PostgreSQL: [https://github.com/postgres/postgres](https://github.com/postgres/postgres)
- pgvector: [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)

## ANN-Benchmark
For benchmarking ANN search performance, we use:
- ANN-Benchmarks: [https://github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks/)

## Datasets
The following datasets are used for evaluation:

- **DBpedia**: [Hugging Face Dataset](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K)
- **Deep**: [Skoltech Deep1B](http://sites.skoltech.ru/compvision/noimi/)
- **GloVe**: [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/)
- **NYTimes**: [UCI Bag of Words](https://archive.ics.uci.edu/ml/datasets/bag+of+words)
- **COCO**: [Skoltech COCO](https://cocodataset.org/#home)

## How to Benchmark
To reproduce the benchmark experiments:

1. **Set up PostgreSQL with pgvector** following the installation guide with the desired branch.
2. **Prepare datasets** using the links above.
3. **Run ANN benchmarks** to evaluate approximate nearest neighbor (ANN) search performance.
4. **Run `CREATE INDEX` command** to measure indexing performance.
