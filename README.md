# PathMoG: Pathway-Centric Modular Graph Network for Multi-Omics Analysis

PathMoG is a graph neural network framework for multi-omics data integration and survival analysis, leveraging pathway knowledge to improve prediction accuracy and interpretability.

## Project Structure

```
gnn/
├── moghet/                # Core model implementation
│   ├── core/             # Data loading and utilities
│   ├── data/             # Raw and processed data (excluded from GitHub)
│   ├── data_processing/  # Data processing scripts
│   ├── evaluation/       # Evaluation metrics
│   ├── inference/        # Inference utilities
│   ├── models/           # Model definitions
│   └── training/         # Training scripts
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

## Core Features

- **Pathway-centric modular design**: Leverages KEGG pathway knowledge to structure the graph architecture
- **Multi-omics integration**: Supports gene expression, mutations, and CNV data
- **Heterogeneous Graph Transformer (HGT)**: Processes different node and edge types
- **Survival prediction**: Built-in Cox proportional hazards model for survival analysis
- **Interpretable results**: Identifies key genes and pathways associated with survival

## Installation

```bash
# Create and activate environment
conda create -n pathmog python=3.9
conda activate pathmog

# Install dependencies
pip install torch torch-geometric
pip install pandas numpy scipy scikit-learn matplotlib seaborn lifelines
```

## Usage

### Data Preparation

The raw data is stored in `moghet/data/raw/` and includes multi-omics data for various cancer types (BRCA, LUAD, GBM, etc.).

### Training

```bash
# Train the model on BRCA dataset
cd moghet
python training/train_hierarchical.py --dataset BRCA --epochs 100
```

### Inference

```bash
# Make predictions using a trained model
cd moghet
python inference/inference.py --model_path models/brca_model.pth --dataset BRCA
```

## Key Files

- `moghet/core/data_loader.py`: Data loading and preprocessing
- `moghet/models/hierarchical_model.py`: Main model definition
- `moghet/training/train_hierarchical.py`: Training script
- `moghet/inference/inference.py`: Inference script
- `moghet/data_processing/build_hetero_graph.py`: Heterogeneous graph construction
- `moghet/evaluation/evaluate_trained_model_auc.py`: Model evaluation

## License

MIT License

## Contact

For questions or collaboration, please contact the project maintainers.
