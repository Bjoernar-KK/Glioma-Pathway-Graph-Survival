# 🧬 Multi-Omics Survival Engine

> A modular deep learning framework for cancer patient survival prediction, built on TCGA data, biological pathway priors, and graph neural networks.

---

## Overview

This project implements a progressive stack of survival models — from a clean universal baseline through to a multi-omics late-fusion architecture — targeting glioblastoma and lower-grade glioma (GBM/LGG) cohorts from The Cancer Genome Atlas (TCGA). Each notebook is a self-contained, independently runnable engine that builds conceptually on the previous one.

The guiding philosophy throughout is **scientific correctness over inflated metrics**: data leakage is explicitly prevented, train/test splits are stratified before any preprocessing, and all architectural choices are annotated with commentary explaining what was wrong in prior baselines and why it was fixed.

---

## Architecture Progression

```
01_Universal_Survival_Engine       ← Foundational Cox + MLP baseline (leakage-free)
         │
         ▼
02_Autoencoder_Engine              ← Denoising autoencoder for latent compression
         │
         ▼
03_Tabular_Survival_Baselines      ← Benchmarks tabular methods; introduces pathway priors
         │
         ▼
04_Pathway_Graph_Fusion_Engine     ← GCN message-passing across biological pathway nodes
         │
         ▼
05_MultiOmics_DNA_Fusion           ← Late-fusion of RNA expression + somatic mutation views
```

---

## Notebooks

### `01_Universal_Survival_Engine.ipynb`
The leakage-corrected foundational engine. Implements chunked low-memory variance filtering over 60,000+ ENSEMBL gene dimensions, strictly isolated StandardScaler fitting on the training partition only, and a baseline Cox proportional hazards head. Establishes reproducibility controls (deterministic seeds across NumPy, PyTorch, and CUDA backends) and a centralized `config.yaml` binding pattern used by all downstream notebooks.

**Key fixes vs. prior baseline:**
- Scaler fitted only on train split — eliminates lookahead bias
- Online variance computation via E[X²] − (E[X])² — avoids full-matrix OOM
- YAML config injection for all paths and hyperparameters

### `02_Autoencoder_Engine.ipynb`
Extends the baseline with a denoising autoencoder (DAE) that compresses the high-dimensional gene expression space into a compact latent bottleneck vector (configurable; default 128-d). The latent representation is then fed to a Cox survival head. Shares the same environment setup and chunked data pipeline as Notebook 01.

### `03_Tabular_Survival_Baselines.ipynb`
Introduces **biological pathway priors** as a structural inductive bias. Partitions the top-5000 variable genes into four canonical cancer pathways:

| Pathway | Genes (approx.) |
|---|---|
| RTK / RAS / PI3K Proliferation | 0–1500 |
| p53 Apoptosis Mechanics | 1500–3000 |
| RB Cell Cycle Checkpoint | 3000–4200 |
| Angiogenesis / Hypoxia Response | 4200–5000 |

Each pathway gets its own `PathwaySubEncoder` (independent MLP), and a cross-talk adjacency matrix defines biologically informed edges. Benchmarks Option A (flat tabular) vs. Option B (pathway-structured) encoding.

### `04_Pathway_Graph_Fusion_Engine.ipynb`
Replaces the static pathway fusion with a **Graph Convolutional Network (GCN)** that performs message-passing across pathway nodes using PyTorch Geometric. The edge index tensor encodes known biological cross-talk (e.g. RTK→p53, RB→RTK). Validated with 5-fold cross-validation; C-index is tracked per fold. Requires `torch-geometric`.

**Architecture:**
```
Patient Gene Vector
       │
  [4 × PathwaySubEncoder]  ← independent per-pathway MLPs
       │
  [GCNConv × 2]            ← message-passing across pathway graph
       │
  [global_mean_pool]
       │
  [Cox Risk Head]           → predicted risk score → C-index
```

### `05_MultiOmics_DNA_Fusion.ipynb`
The most advanced engine. Fuses two omic views via a **learned attention gating mechanism**:

```
RNA-Seq expression  →  PathwayGCN  →  RNA embedding [128]
                                              │
                              ViewAttentionFusion  →  Cox Head
                                              │
DNA somatic mutations  →  MutationPathwayEncoder  →  DNA embedding [128]
```

RNA data: HTSeq counts from TCGA GBM/LGG (GDC API, no login required).
DNA data: Masked Somatic Mutations MAF → binarised gene × patient matrix.
Both views are pathway-partitioned and aligned to the same patient cohort before fusion. A synthetic data synthesis fallback is included for offline development and testing.

---

## Setup

### Requirements

```bash
pip install torch torchvision
pip install torch-geometric          # notebooks 03–05
pip install lifelines scikit-learn pandas numpy matplotlib pyyaml
```

GPU is optional but recommended for Notebooks 04–05.

### Configuration

All notebooks read from a shared `config.yaml`. Create it at your preferred path and point `CONFIG_PATH` in each notebook to it:

```yaml
project:
  cancer_type: GBM

data:
  processed_dir: /path/to/data
  clinical_file: tcga_expression_survival.csv

model:
  latent_dim: 128

preprocessing:
  variance_filter_top_n: 5000
```

### Data

TCGA GBM/LGG RNA-Seq and mutation data can be downloaded programmatically via the NCI GDC API (Cell 2 of Notebook 05 handles this automatically). For offline development, Notebook 05 includes a high-fidelity synthetic data engine that mimics TCGA cohort statistics.

---

## Key Design Decisions

**Leakage prevention** — The StandardScaler is fitted exclusively on the training partition. This is enforced at the architectural level, not as an afterthought.

**Chunked variance filtering** — Gene selection uses online computation (running sum + sum-of-squares) to avoid loading a 60,000-column matrix into RAM.

**Pathway priors as inductive bias** — Rather than treating all genes as exchangeable features, the model respects known cancer biology by assigning genes to canonical oncogenic pathways and learning pathway-level representations.

**Graph message-passing** — Biological cross-talk between pathways (e.g. RTK↔p53, RB→RTK) is encoded as a graph, allowing the GCN to propagate information across pathway boundaries in a structure-aware way.

**Multi-omics late fusion** — RNA and DNA views are encoded independently before fusion, allowing each modality to develop its own representation before the attention gate weights their relative contribution.

---

## Evaluation

Primary metric: **Concordance Index (C-index)** via `lifelines.utils.concordance_index`.

Notebooks 04 and 05 implement **5-fold cross-validation** with per-fold C-index tracking. A random risk assignment gives C-index ≈ 0.5; a clinically useful model typically exceeds 0.65 on held-out TCGA data.

---

## Limitations & Honest Caveats

- Sample sizes in TCGA GBM/LGG cohorts are small (n ≈ 150–300 after overlap filtering), which limits the complexity of models that can be reliably trained without overfitting.
- Pathway gene boundaries are defined by index ranges over the top-5000 variable genes, not curated gene set databases (e.g. MSigDB). Replacing these with literature-validated pathway memberships would strengthen biological interpretability.
- The DNA mutation encoder uses a binary presence/absence representation. Mutation type, variant allele frequency, and functional consequence are not currently encoded.
- Survival times and event indicators are taken at face value from the TCGA clinical manifest; right-censoring patterns are not formally audited.

---

## Citation / Data Sources

- TCGA GBM/LGG cohort: [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/)
- Survival analysis: [lifelines](https://lifelines.readthedocs.io/)
- Graph neural networks: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

---

## License

MIT
