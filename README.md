# CAPID

This repository contains code for working with **CAPID**, a dataset and pipeline for context-aware PII detection and relevance estimation.


## Setup

### 1. Create a Conda environment

We recommend using **Python 3.10**.

```bash
conda create -n capid-env python=3.10
conda activate capid-env
```
### 2. Install the package

Upgrade core tooling and install all required Python dependencies from `requirements.txt`:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Install the package

```bash
pip install -e .
```

## Running the pipeline

To try out the pipeline, navigate to the package directory and run the pipeline script:

```bash
cd src/capid
python pipeline.py
```

## Data

Training and test data are located in the data/ directory:

```bash
data/train.jsonl
data/test.jsonl
data/reddit.jsonl
```

These files are used for training and evaluation, respectively.
