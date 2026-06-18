## 🧬 Advanced Feature Engineering: Structural Pathway Slicing

To move beyond generic multi-modal concatenation, the architecture implements a **graph-aware preprocessing pipeline** in Cells 2-4. Instead of exposing the deep learning head to a flat feature vector, patient records are dynamically segmented into localized functional modules.

### 1. Structural Graph Topology (Cell 2)
* **Pathway Graph Nodes:** 4 distinct biological pathways are registered as macro-nodes.
* **Cross-Talk Edge Map:** An adjacency list of shape `[2, 8]` defines 8 explicit interaction pathways representing biological cross-talk and signal transduction boundaries.

### 2. Pre-trained Weight Integration (Cell 3)
* Four independent local sub-networks are established. Each pathway sub-network inherits pre-trained weights from the unsupervised **Version 2 DAE**, ensuring that local feature representations are denoised and optimized prior to graph collation.

### 3. Custom Batch Collation Engine (Cell 4)
Using a custom PyTorch Dataset (`PatientGraphDataset`) and an explicit collation function (`graph_collate_fn`), raw patient feature vectors are systematically sliced into pathway-specific tensors:

* **Slicing Mechanism:** Continuous indices are mapped dynamically to isolate pathway-specific coordinates.
* **Collation Output:** Batches are organized into a pathway-keyed dictionary mapping directly to downstream graph convolutional layers:

$$\text{Batch Output} = \{ \text{Pathway Name} : \text{Tensor}_{\text{Batch} \times \text{Slice Length}} \}$$

This customized structural slicing layout directly prepares the multi-modal patient cohort for True Pathway Graph Fusion, shifting the framework from generic numerical curve fitting to biologically-informed survival prediction.
