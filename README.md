# Query Reformulation Baselines

This directory contains implementations of 5 query reformulation baselines: **MuGI**, **LameR**, **QA-EXPAND**, **GenQR-Ensemble**, and **FlanQR**.

All baselines support both **local execution** (CPU/MPS for M1 Macs) and **GPU execution** (vLLM on SLURM clusters).

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Baselines Overview](#baselines-overview)
- [Running Locally (CPU/MPS)](#running-locally-cpumps)
- [Running on SLURM (GPU)](#running-on-slurm-gpu)
- [Input Requirements](#input-requirements)
- [Output Format](#output-format)
- [Models Used](#models-used)

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch transformers vllm pandas tqdm

# For local execution on M1 Macs, ensure PyTorch MPS support is available
```

### Directory Structure

```
baselines/
├── mugi/
│   ├── mugi.py              # GPU version (vLLM)
├── lamer/
│   └── lamer.py             # Unified (supports both --env gpu/local)
├── qa-expand/
│   └── qa-expand.py         # Unified (supports both --env gpu/local)
├── gen-qr-ensemble/
│   └── gen-qr-ensemble.py   # Unified (supports both --env gpu/local)
├── flan-qr/
│   └── flan-qr.py           # Unified (supports both --env gpu/local)
└── data/
    └── trecdl2019/
        ├── queries.tsv
        ├── collection.tsv
        └── trec.run
```

---

## Baselines Overview

| Baseline | Description | LLM Calls | Output |
|----------|-------------|-----------|--------|
| **MuGI** | Multi-Text Generation Integration - generates pseudo-documents and concatenates with original query | 5 (one per doc) | Expanded query = (Q × 5) + 5 pseudo-docs |
| **LameR** | LLM-Augmented Multi-stage Retrieval - generates pseudo-passages from BM25 context | 1 (batch) | Expanded query = Q + 5 interleaved passages |
| **QA-EXPAND** | Multi-Question Answer Generation - generates sub-questions, answers, and refines | 3 (sub-q, ans, refine) | Expanded query = (Q × 3) + refined answers |
| **GenQR-Ensemble** | Ensemble of 10 instruction variants for keyword generation | 10 (one per instruction) | Expanded query = (Q × 5) + all keywords |
| **FlanQR** | Single instruction-based expansion using Qwen | 1 | Expanded query = (Q × 5) + expansion text |

---

## Running Locally (CPU/MPS)

### 1. MuGI (Local)

```bash
cd baseline/mugi

python3 mugi_local.py \
  --queries ../data/trecdl2019/queries.tsv \
  --output mugi_trecdl2019_local.tsv \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num_docs 5 \
  --adaptive_times 5
```

**Parameters:**
- `--queries`: Path to query TSV file (qid, query)
- `--output`: Output TSV file path
- `--model`: HuggingFace model name (default: Qwen/Qwen2.5-1.5B-Instruct)
- `--num_docs`: Number of pseudo-documents to generate (default: 5)
- `--adaptive_times`: Query repetition weight (default: 5)

---

### 2. LameR (Local)

```bash
cd baseline/lamer

python3 lamer.py \
  --queries ../data/trecdl2019/queries.tsv \
  --collection ../data/trecdl2019/collection.tsv \
  --bm25_run ../data/trecdl2019/trec.run \
  --output lamer_trecdl2019_local.tsv \
  --env local \
  --debug
```

**Parameters:**
- `--queries`: Path to query TSV file
- `--collection`: Path to collection file (TSV or JSONL)
- `--bm25_run`: Path to pre-computed BM25 run file (TREC format)
- `--output`: Output TSV file path
- `--env`: `local` for CPU/MPS, `gpu` for vLLM
- `--num_passages`: Number of pseudo-passages (default: 5)
- `--debug`: Enable verbose logging

---

### 3. QA-EXPAND (Local)

```bash
cd baseline/qa-expand

python3 qa-expand.py \
  --queries ../data/trecdl2019/queries.tsv \
  --output qa_expand_trecdl2019_local.tsv \
  --env local \
  --debug
```

**Parameters:**
- `--queries`: Path to query TSV file
- `--output`: Output TSV file path
- `--model`: Model name (default: Qwen2.5-1.5B for local)
- `--num_subquestions`: Number of sub-questions (default: 3)
- `--repeat_query_weight`: Query repetition (default: 3)
- `--env`: `local` for CPU/MPS, `gpu` for vLLM
- `--debug`: Enable verbose logging

---

### 4. GenQR-Ensemble (Local)

```bash
cd baseline/gen-qr-ensemble

python3 gen-qr-ensemble.py \
  --queries ../data/trecdl2019/queries.tsv \
  --output genqr_trecdl2019_local.tsv \
  --env local \
  --debug
```

**Parameters:**
- `--queries`: Path to query TSV file
- `--output`: Output TSV file path
- `--model`: Model name (default: Qwen2.5-1.5B for local)
- `--repeat_query_weight`: Query repetition (default: 5)
- `--env`: `local` for CPU/MPS, `gpu` for vLLM
- `--debug`: Enable verbose logging

---

### 5. FlanQR (Local)

```bash
cd baseline/flan-qr

python3 flan-qr.py \
  --queries ../data/trecdl2019/queries.tsv \
  --output flanqr_trecdl2019_local.tsv \
  --env local \
  --debug
```

**Parameters:**
- `--queries`: Path to query TSV file
- `--output`: Output TSV file path
- `--model`: Model name (default: Qwen2.5-1.5B for local)
- `--repeat_query_weight`: Query repetition (default: 5)
- `--env`: `local` for CPU/MPS, `gpu` for vLLM
- `--debug`: Enable verbose logging

---

## Running on SLURM (GPU)

### Setting Up Environment (Zeus/Artemis)

```bash
# Set HuggingFace cache directory
export HF_HOME=/mnt/data/$USER/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/data/$USER/hf_cache

# Create cache directory
mkdir -p /mnt/data/$USER/hf_cache
```

---

### 1. MuGI (GPU)

Create `run_mugi.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=mugi
#SBATCH --output=mugi_%j.out
#SBATCH --error=mugi_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Set cache directories
export HF_HOME=/mnt/data/$USER/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/data/$USER/hf_cache

# Activate environment
source /mnt/data/$USER/miniconda/bin/activate ollama-py

# Run MuGI
cd /mnt/data/$USER/llm-query-reformulation/baseline/mugi

python3 mugi.py \
  --queries ../data/trecdl2019/queries.tsv \
  --output mugi_trecdl2019_gpu.tsv \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num_docs 5 \
  --adaptive_times 5
```

Submit:
```bash
sbatch run_mugi.sh
```

---

### 2. LameR (GPU)

Create `run_lamer.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=lamer
#SBATCH --output=lamer_%j.out
#SBATCH --error=lamer_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Set cache directories
export HF_HOME=/mnt/data/$USER/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/data/$USER/hf_cache

# Activate environment
source /mnt/data/$USER/miniconda/bin/activate ollama-py

# Run LameR
cd /mnt/data/$USER/llm-query-reformulation/baseline/lamer

python3 lamer.py \
  --queries ../data/trecdl2019/queries.tsv \
  --collection ../data/trecdl2019/collection.tsv \
  --bm25_run ../data/trecdl2019/trec.run \
  --output lamer_trecdl2019_gpu.tsv \
  --env gpu
```

Submit:
```bash
sbatch run_lamer.sh
```

---

### 3. QA-EXPAND (GPU)

Create `run_qa_expand.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=qa_expand
#SBATCH --output=qa_expand_%j.out
#SBATCH --error=qa_expand_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Set cache directories
export HF_HOME=/mnt/data/$USER/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/data/$USER/hf_cache

# Activate environment
source /mnt/data/$USER/miniconda/bin/activate ollama-py

# Run QA-EXPAND
cd /mnt/data/$USER/llm-query-reformulation/baseline/qa-expand

python3 qa-expand.py \
  --queries ../data/trecdl2019/queries.tsv \
  --output qa_expand_trecdl2019_gpu.tsv \
  --env gpu
```

Submit:
```bash
sbatch run_qa_expand.sh
```

---

### 4. GenQR-Ensemble (GPU)

Create `run_genqr.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=genqr
#SBATCH --output=genqr_%j.out
#SBATCH --error=genqr_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Set cache directories
export HF_HOME=/mnt/data/$USER/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/data/$USER/hf_cache

# Activate environment
source /mnt/data/$USER/miniconda/bin/activate ollama-py

# Run GenQR-Ensemble
cd /mnt/data/$USER/llm-query-reformulation/baseline/gen-qr-ensemble

python3 gen-qr-ensemble.py \
  --queries ../data/trecdl2019/queries.tsv \
  --output genqr_trecdl2019_gpu.tsv \
  --env gpu
```

Submit:
```bash
sbatch run_genqr.sh
```

---

### 5. FlanQR (GPU)

Create `run_flanqr.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=flanqr
#SBATCH --output=flanqr_%j.out
#SBATCH --error=flanqr_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Set cache directories
export HF_HOME=/mnt/data/$USER/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/data/$USER/hf_cache

# Activate environment
source /mnt/data/$USER/miniconda/bin/activate ollama-py

# Run FlanQR
cd /mnt/data/$USER/llm-query-reformulation/baseline/flan-qr

python3 flan-qr.py \
  --queries ../data/trecdl2019/queries.tsv \
  --output flanqr_trecdl2019_gpu.tsv \
  --env gpu
```

Submit:
```bash
sbatch run_flanqr.sh
```

---

## Input Requirements

### Query File Format (`.tsv`)

```tsv
qid	query
156493	do goldfish grow
1110199	what is wifi vs bluetooth
```

### Collection File Format (`.tsv` for MS MARCO)

```tsv
docid	text
7187158	The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was...
```

### BM25 Run File Format (TREC format)

```
qid Q0 docid rank score run_name
156493 Q0 7187158 1 23.5 bm25
156493 Q0 2345678 2 22.1 bm25
```

---

## Output Format

### MuGI Output

```tsv
qid	pseudo_doc_1	pseudo_doc_2	...	pseudo_doc_5	enhanced_query
156493	Goldfish grow...	They can reach...	...	...	do goldfish grow do goldfish grow...
```

### LameR Output

```tsv
qid	pseudo_passage_1	...	pseudo_passage_5	expanded_query
156493	Goldfish growth depends...	...	...	do goldfish grow passage1 do goldfish grow passage2...
```

### QA-EXPAND Output

```tsv
qid	subquestion_1	subquestion_2	subquestion_3	answer_1	answer_2	answer_3	expanded_query
156493	Do goldfish change size?	How do factors affect growth?	What is lifespan?	No, goldfish...	Several factors...	Average lifespan...	do goldfish grow do goldfish grow do goldfish grow answer1 answer2 answer3
```

### GenQR-Ensemble Output

```tsv
qid	keywords_1	keywords_2	...	keywords_10	expanded_query
156493	fish, aquarium, tank	growth rate, size	...	...	do goldfish grow do goldfish grow... keywords
```

### FlanQR Output

```tsv
qid	expansion_text	expanded_query
156493	growth rate, aquarium care, fish size	do goldfish grow do goldfish grow... expansion_text
```

---

## Models Used

| Environment | Model | Size | Speed |
|-------------|-------|------|-------|
| **Local** (CPU/MPS) | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B params | Slow (~5-10 min/query) |
| **GPU** (vLLM) | `Qwen/Qwen2.5-7B-Instruct` | 7B params | Fast (~10-30 sec/query) |

---

## Monitoring Jobs (SLURM)

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f <baseline>_<job_id>.out

# View error logs
tail -f <baseline>_<job_id>.err

# Cancel job
scancel <job_id>
```

---

## Troubleshooting

### Issue: `PermissionError` on cache directory

**Solution:**
```bash
mkdir -p /mnt/data/$USER/hf_cache
export HF_HOME=/mnt/data/$USER/hf_cache
```

### Issue: Out of memory on local execution

**Solution:** Use smaller model:
```bash
--model Qwen/Qwen2.5-0.5B-Instruct
```

### Issue: vLLM fails to load model

**Solution:** Lower GPU memory utilization:
```python
# In the script, modify:
gpu_memory_utilization=0.5  # instead of 0.7
```

### Issue: Slow local execution

**Solution:** Use GPU mode on Zeus/Artemis or reduce dataset size for testing.

---

## Citation

If you use these baselines, please cite the original papers:

- **MuGI**: Multi-Text Generation Integration for Information Retrieval
- **LameR**: LLM-Augmented Multi-stage Retrieval
- **QA-EXPAND**: Multi-Question Answer Generation for Enhanced Query Expansion
- **GenQR-Ensemble**: Ensemble of Instruction-Based Query Reformulation
- **FlanQR**: Flan-T5 Query Reformulation

---

## Contact

For issues or questions, please open an issue in the repository.

