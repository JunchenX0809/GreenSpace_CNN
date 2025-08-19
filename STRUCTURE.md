# Repository Structure - GreenSpace CNN

## 📁 Complete Directory Layout

```
GreenSpace_CNN/
├── 📄 README.md                     # Project overview and setup
├── 📄 LICENSE                       # License file
├── 📄 requirements.txt              # Python dependencies (TensorFlow/Keras stack)
├── 📄 STRUCTURE.md                  # This file - repository structure guide
│
├── 📁 data/                         # Data directory (see data/README.md)
│   ├── 📄 README.md                 # Data structure documentation
│   ├── 📁 raw/                      # Original, immutable data
│   │   ├── 📁 images/               # Raw satellite/aerial images
│   │   └── 📄 survey_responses.csv  # Raw survey responses
│   ├── 📁 processed/                # Cleaned and processed data
│   │   ├── 📁 images/               # Preprocessed images
│   │   ├── 📄 labels.csv            # Processed labels for training
│   │   └── 📁 splits/               # Train/val/test splits
│   ├── 📁 external/                 # External datasets (optional)
│   └── 📁 interim/                  # Intermediate processing files
│
├── 📁 notebooks/                    # Jupyter notebooks (main workflow)
│   ├── 📓 01_data_exploration.ipynb      # Survey data analysis & visualization
│   ├── 📓 02_data_preprocessing.ipynb    # Data cleaning & preparation
│   ├── 📓 03_model_training.ipynb        # Multi-task CNN training
│   ├── 📓 04_model_evaluation.ipynb      # Model evaluation & analysis
│   └── 📓 05_prediction_demo.ipynb       # Interactive prediction demo
│
├── 📁 src/                          # Source code modules
│   ├── 📄 utils.py                  # Utility functions (data processing, config)
│   └── 📄 models.py                 # Multi-task CNN model definitions
│
├── 📁 config/                       # Configuration files
│   └── 📄 model_config.yaml         # Model, training, and data configuration
│
├── 📁 models/                       # Trained model artifacts (created during training)
│   └── 📄 best_model.h5             # Best trained model checkpoint
│
├── 📁 logs/                         # Training logs and outputs (created during training)
│   ├── 📄 training_log.csv          # Training metrics log
│   └── 📁 tensorboard/              # TensorBoard visualization files
│
└── 📁 outputs/                      # Results and predictions (created during inference)
    ├── 📄 predictions.csv           # Model predictions on new data
    └── 📁 visualizations/           # Generated plots and figures
```

## 🚀 Getting Started Workflow

### 1. **Setup Environment**
```bash
pip install -r requirements.txt
```

### 2. **Data Preparation**
- Place your survey responses in `data/raw/survey_responses.csv`
- Place images in `data/raw/images/`
- Run `notebooks/01_data_exploration.ipynb` to understand your data

### 3. **Data Processing**
- Run `notebooks/02_data_preprocessing.ipynb` to:
  - Clean and aggregate survey responses
  - Process images and create data splits
  - Set up TensorFlow data pipelines

### 4. **Model Training**
- Configure model settings in `config/model_config.yaml`
- Run `notebooks/03_model_training.ipynb` to:
  - Build multi-task CNN architecture
  - Train with proper callbacks and monitoring
  - Save best model to `models/`

### 5. **Evaluation**
- Run `notebooks/04_model_evaluation.ipynb` to:
  - Compute comprehensive metrics
  - Analyze model performance per task
  - Generate visualizations

### 6. **Predictions**
- Use `notebooks/05_prediction_demo.ipynb` for:
  - Interactive predictions on new images
  - Batch processing and CSV export

## 🏗️ Architecture Highlights

### **Multi-task Model Design**
```python
# Three prediction heads:
├── Structured Rating (Regression): 0-1 scale
├── Binary Features (Multi-binary): 6 greenspace features  
└── Shade Level (Categorical): None/Some/Abundant
```

### **Technology Stack**
- **Framework**: TensorFlow 2.15+ with Keras API
- **Backbone**: EfficientNet/ResNet with ImageNet pretraining
- **Data Pipeline**: tf.data for efficient loading
- **Visualization**: matplotlib, seaborn, plotly
- **Experiment Tracking**: TensorBoard + optional Weights & Biases

### **Key Features**
- ✅ **Notebook-driven workflow** for accessibility
- ✅ **Multi-task learning** with shared CNN backbone
- ✅ **Configurable architecture** via YAML files
- ✅ **Comprehensive evaluation** with per-task metrics
- ✅ **Interactive demos** for model exploration
- ✅ **Geospatial support** for satellite imagery

## 📝 File Naming Conventions

- **Notebooks**: Numbered prefix (01_, 02_, etc.) for workflow order
- **Data files**: Descriptive names with underscores (survey_responses.csv)
- **Images**: Keep original names or use consistent pattern
- **Models**: Version suffix or timestamp for tracking
- **Config**: Purpose-specific names (model_config.yaml)

## 🔧 Customization Points

1. **Model Architecture** (`config/model_config.yaml`):
   - Change backbone CNN (EfficientNet, ResNet, etc.)
   - Adjust task weights and loss functions
   - Modify data augmentation settings

2. **Data Processing** (`src/utils.py`):
   - Customize multi-rater aggregation methods
   - Add new preprocessing steps
   - Extend label encoding functions

3. **Training Pipeline** (`notebooks/03_model_training.ipynb`):
   - Experiment with different optimizers
   - Add custom callbacks and metrics
   - Implement advanced training strategies

This structure balances **accessibility** (Jupyter notebooks) with **organization** (modular code) and **reproducibility** (configuration files).
