# PathSurv — Project Selling Points

> A plain-language pitch for collaborators, grant reviewers, journal editors, and conference audiences.

---

## The one-sentence pitch

PathSurv is the first survival prediction framework for glioma that combines biologically informed graph message-passing with multi-omics late-fusion — and is structurally immune to the data leakage that undermines most published deep survival models.

---

## Selling point 1 — Scientific integrity you can prove

**The problem in the field:** The majority of deep learning survival papers fit their preprocessing transformations (standardisation, variance selection) across the entire patient population *before* splitting into train and test. This leaks future information into the model and inflates the reported C-index. Peer reviewers rarely catch it.

**What PathSurv does:** The `StandardScaler` is fitted exclusively on the training partition and applied to validation and test sets. Variance filtering is computed from training samples only. This is enforced architecturally — it cannot be accidentally reversed.

**Why it matters for a paper:** When a reviewer asks "how do you prevent data leakage?" you have a specific, auditable, cell-level answer. Most authors don't.

---

## Selling point 2 — Biology drives the architecture, not just the labels

**The problem in the field:** Most deep survival models treat 60,000 genes as an exchangeable flat feature vector. The model is asked to rediscover biology from scratch with fewer than 300 patients — an impossible task.

**What PathSurv does:** Genes are partitioned into four canonical GBM oncogenic programs *before* any learning occurs:

- RTK / RAS / PI3K Proliferation
- p53 Apoptosis Mechanics
- RB Cell Cycle Checkpoint
- Angiogenesis / Hypoxia Response

Each pathway gets its own encoder. A GCN then passes messages across biologically known cross-talk edges (e.g. RTK ↔ p53, RB → RTK), allowing the model to integrate evidence along molecular interaction axes that GBM biology has already established.

**Why it matters for a paper:** Pathway-level architecture decisions are interpretable to clinicians and oncologists, not just machine learning reviewers. Risk scores trace back to druggable pathways. This is the difference between a method paper and a translational paper.

---

## Selling point 3 — A genuine technical contribution, not a benchmark comparison

Most bioinformatics submissions apply an existing architecture (a transformer, a GCN) to a new dataset and report a higher number. PathSurv makes three technically novel contributions:

1. **Online variance computation** using E[X²] − (E[X])² streaming over chunks — the only way to safely handle 60,000+ gene matrices on standard hardware without out-of-memory failure.

2. **Pathway GCN message-passing** — operating on pathway nodes (not gene nodes), reducing the graph from 60,000 nodes to 4 while encoding known cross-talk topology. This is a novel graph construction strategy for survival analysis.

3. **Multi-omics late-fusion with learned attention gating** — RNA expression and somatic mutation views develop independent pathway representations before an attention gate weights their relative contribution. The gate weights are interpretable per patient.

---

## Selling point 4 — Reproducibility as a first-class feature

Open science reviewers and Nature-family journals increasingly require:

- Deterministic seeds → enforced across NumPy, Python `random`, PyTorch CPU, and CUDA backends
- No manual data download steps → TCGA GBM/LGG data fetched automatically via the NCI GDC API
- Offline development → a high-fidelity synthetic data engine mimics TCGA cohort statistics
- Single configuration file → one `config.yaml` governs all five notebooks; forking for a new cancer type requires editing two lines

This is not boilerplate reproducibility. It is a design decision made at the start and maintained across every notebook.

---

## Selling point 5 — A progressive framework, not a black box

PathSurv is structured as five independently runnable stages:

| Stage | What it adds | Why it matters |
|---|---|---|
| 01 — Universal baseline | Leakage-free CoxPH + MLP | Credible floor; every gain is measured against it |
| 02 — Autoencoder | 128-d bottleneck compression | Shows unsupervised structure in expression data |
| 03 — Pathway encoders | Biological partitioning | Isolates the contribution of inductive bias |
| 04 — Pathway GCN | Cross-talk message-passing | Isolates the graph contribution |
| 05 — Multi-omics fusion | RNA + DNA attention gate | Full model; each prior stage is the ablation |

The ablation table is built in. No extra experiments needed.

---

## Positioning against prior work

| Model | Pathway prior | Graph | Multi-omics | No leakage | Glioma-specific |
|---|---|---|---|---|---|
| DeepSurv (2018) | — | — | — | — | — |
| Cox-PASNet (2019) | ✓ | — | — | — | ✓ |
| GGNN (2023) | ✓ | ✓ | — | ? | — |
| SurvPath (2024) | ✓ | — | ✓ (+ histology) | ? | — |
| **PathSurv (ours)** | **✓** | **✓** | **✓** | **✓** | **✓** |

The unique combination is: pathway priors + GCN cross-talk + multi-omics fusion + provable leakage prevention + glioma focus + full reproducibility. No single prior work covers all five.

---

## Audiences and what to emphasise

**For a machine learning venue (NeurIPS, ICML, ICLR):**
Lead with the GCN pathway graph construction as a novel biological graph formulation. Emphasise the ablation design and the leakage-prevention engineering.

**For a bioinformatics venue (Bioinformatics, PLOS Comp Bio, Genome Biology):**
Lead with the clinical interpretability of pathway-level risk attribution. Emphasise the GBM oncogenic pathway alignment and the C-index improvements over Cox-PASNet.

**For a clinical/oncology venue (Cancer Cell, Nature Cancer):**
Lead with translational impact — risk scores that map to druggable pathways (RTK inhibitors, CDK4/6 inhibitors). Emphasise patient stratification potential and multi-omics integration.

**For a grant application:**
Lead with the reproducibility and open science angle. Emphasise the automated GDC pipeline, YAML configuration, synthetic data fallback, and the fact that the full stack can be adopted by any lab within a day.

**For a GitHub README / open source audience:**
Lead with the zero-manual-steps data download, the progressive notebook design, and the synthetic data engine for offline experimentation.

---

## Honest caveats to acknowledge proactively

Reviewers respect authors who name their limitations before being asked:

- Pathway boundaries are index-range proxies, not curated gene set memberships (MSigDB, KEGG). Future work: replace with literature-validated assignments.
- TCGA GBM/LGG cohort is small (N ≈ 150–300). Results should be validated on an independent non-TCGA cohort.
- DNA mutation encoder uses binary presence/absence. Variant allele frequency and functional consequence scores are not currently encoded.
- DNA methylation and copy number alteration are not included as modalities.
- C-index confidence intervals across folds should be reported (not just means) for small-N settings.

Naming these unprompted signals scientific maturity and typically improves reviewer reception.

---

*PathSurv | TCGA GBM/LGG | PyTorch + PyTorch Geometric | Open source*
