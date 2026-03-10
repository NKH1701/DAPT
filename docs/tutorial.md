This tutorial walks you through the complete DAPT pipeline, from processing raw single-cell transcriptomics data to evaluating the model's predictions.

## ⬇️ Installation

From Source

```bash
git clone https://github.com/NKH1701/DAPT.git
```

## 📂 Data Loading & Preprocessing


Before training the model, we need to process the raw data and construct our biological knowledge graphs (Gene Ontology and Gene Co-expression).

**1. Process the Single-Cell Data**

The first step in the pipeline is to read the raw .h5ad single-cell dataset (e.g., K562, Norman, Adamson) and generates crucial mapping dictionaries. 

These dictionaries map human-readable gene names and perturbation conditions to internal tensor indices, which are then saved as a .pkl metadata file for the Constellation orchestrator to use later.

```python
# Define the target dataset
dataset_name = "k562"  # Options: norman, adamson, dixit, rpe1, k562

# 1. Read the raw Single-Cell AnnData file
path_read_h5ad = f"../data_raw/{dataset_name}.h5ad"
adata = sc.read_h5ad(path_read_h5ad)

# ... [Internal processing to extract condition/control mappings] ...

# 2. Package all mappings into a metadata dictionary
meta = {
    "map_cond_responses": map_cond_responses,
    "map_pert_perturbation": map_pert_perturbation,
    "map_perturbation_pert": map_perturbation_pert,
    "map_gene_loc_gene_name": map_gene_loc_gene_name,
    "map_cond_non_zero_gene_locs_asc": map_cond_non_zero_gene_locs_asc,
    "map_cond_non_drop_gene_locs_asc": map_cond_non_drop_gene_locs_asc,
    # ... and rank mappings ...
}

# 3. Export the processed metadata for model training
path_save = "../data_processed"
os.makedirs(path_save, exist_ok=True)

with open(os.path.join(path_save, f"meta_{dataset_name}.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"Successfully processed and saved metadata for {dataset_name}!")
```

**2. Construct the Biological Graphs**

To provide our Graph Neural Network (GNN) with the necessary structural priors, we process raw gene co-expression data and Gene Ontology (GO) terms.

This step generates the explicit edge weights and node lists that the our Constellation manager uses to build the graph topology.

We first read a raw file that containing our high-mean target genes and structures them into a clean JSON index

```python
# 1. Read the source gene list
path_read = "../data_raw/genes_with_hi_mean.npy"
genes = np.load(path_read, allow_pickle=True)

print(f"Shape of the data: {genes.shape}")
print(f"Content preview: {genes[:3]}")

# 2. Save the processed interaction genes
path_save = "../data_processed"
os.makedirs(path_save, exist_ok=True)

with open(os.path.join(path_save, "interaction_genes.json"), "w", encoding="utf-8") as f:
    json.dump(genes.tolist(), f)
```

Then we can handles the edge structures. We reads the raw Gene Ontology mappings, validates that there are no missing values, checks for symmetry, and eventually exports the edge list alongside vital perturbation dictionaries.

```python
# 1. Read the source CSV containing source-target GO relations
path_read = "../data_raw/go_essential_all.csv"
go_raw = pd.read_csv(path_read)

# 2. Verify graph properties
print("First 5 edges in the GO graph:")
print(go_raw.head())

print(f"Does the DataFrame have any NaN values? {go_raw.isna().values.any()}")
print(f"Number of unique perturbations: {len(pd.unique(go_raw[['source', 'target']].values.ravel('K')))}")

# 3. Check Edge Weights (Importance)
importance_min = go_raw['importance'].min()
importance_max = go_raw['importance'].max()
print(f"The range of values for the 'importance' column is: {importance_min:0.4f} to {importance_max:0.4f}")

# ... [Filtering logic to generate go_new and mapping dictionaries] ...

# 4. Export the clean graph and mappings
go_new.to_csv(os.path.join(path_save, "go.csv"), index=False)

with open(os.path.join(path_save, "map_perturbation_pert.json"), "w", encoding="utf-8") as f:
    json.dump(map_perturbation_pert, f)
```

## 🗄️ The Data Manager 

Once our data is preprocessed, we load it into our system using the Constellation class. This acts as a Facade, hiding the complexity of the internal data mappings, biological state storage , and graph topology.

```python
# Initialize the Constellation orchestrator
# This automatically loads all mappings, split definitions, and graph structures
constellation = Constellation(config_path="configs/experiment_config.yaml")

# Retrieve the PyTorch DataLoaders (Train, Validation, Test)
# The 'Postman' class handles the safe batching of the AnnData objects
dataloaders = constellation.get_loaders(batch_size=32)
loader_trn = dataloaders["loader_trn"]
loader_val = dataloaders["loader_val"]
loader_tst = dataloaders["loader_tst"]

# Retrieve the knowledge graph edges for the GNN
edges_p = dataloaders["edges_p"]
edges_g = dataloaders["edges_g"]
```

## 🏗️ Model Initialization

With our data ready, we can initialize the core neural network model. The model requires the configuration dictionary generated by the Constellation manager to ensure the input dimensions perfectly match the dataset.

```python
# Initialize the Model 
# (Note: 'config_model' defines the GNN layers, autoencoder dims, etc.)
config_model = constellation.get_model_config()
model = Gears(**config_model)

# Move the model to the GPU for accelerated training
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

## 📝 Training & Evaluation

The training loop iterates through the data delivered by the data loaders, optimizes the loss, and evaluates the predictions against the withheld test set.

We evaluate the framework using a comprehensive suite of bioinformatics metrics including Precision, MSE, Pearson correlation, etc.

```python
# Execute the standard training and evaluation pipeline
# This automatically handles the GNN message passing and autoencoder reconstruction
results = model.exp_standard(
    loader_trn=loader_trn, 
    loader_val=loader_val, 
    loader_tst=loader_tst, 
    device=device, 
    constellation=constellation
)

# Print final evaluation metrics
print(f"Test MSE: {results['mse_loss']:.4f}")
print(f"Precision @ 20: {results['precision_at_k']:.4f}")
print(f"Directionality Accuracy: {results['directionality']:.4f}")
```

All experimental results are seamlessly appended to a csv file. This allows for easy tracking of model performance across different hyperparameter configurations and simulation settings.
