# Deep Neural Network Structure Analysis

A comprehensive toolkit for extracting, processing, and analyzing the architectural patterns of deep neural network models.

## Project Overview

This project provides a pipeline for extracting and analyzing the internal structure of neural network architectures from Hugging Face's model hub. The pipeline consists of three main stages:

1. **Model Extraction**: Extract the internal architecture of various neural network models from Hugging Face.
2. **Data Processing**: Process the extracted data, including renaming and deduplication.
3. **Architecture Analysis**: Analyze the processed data to identify patterns, common structures, and task-specific architectural components.

## Directory Structure

```
Deep_NN_Structure/
├── model_extract/            # Model extraction scripts
│   ├── config.py             # Configuration settings
│   ├── utils.py              # Utility functions
│   ├── model_loader.py       # Model loading functions
│   ├── structure_extractor.py # Structure extraction functions
│   ├── main.py               # Main extraction logic
│   └── run.py                # Entry point script
├── model_architecture/       # Extracted model architectures
│   ├── raw/                  # Raw extracted model files
│   └── processed/            # Deduplicated and processed files
├── data_processed/           # Data processing scripts
│   ├── rename_csv_files.py   # File renaming script
│   └── deduplication.py      # Deduplication script
├── analysis/                 # Analysis scripts
│   ├── config.py             # Analysis configuration
│   ├── data_loader.py        # Data loading functions
│   ├── basic_analysis.py     # Basic analysis functions
│   ├── advanced_analysis.py  # Advanced analysis with visualizations
│   ├── architecture_analyzer.py # Main analysis logic
│   └── run.py                # Entry point script
├── analysis_result/          # Analysis results
│   ├── images/               # Visualizations and plots
│   ├── csv_data/             # Processed CSV data
│   ├── reports/              # Analysis reports
│   └── statistics_txt/       # Statistical summaries
├── run_pipeline.py           # Full pipeline orchestrator
└── deep_nn_structure_env.yml # Environment file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Deep_NN_Structure.git
   cd Deep_NN_Structure
   ```

2. Create the conda environment:
   ```bash
   conda env create -f deep_nn_structure_env.yml
   conda activate deep_nn_structure
   ```

## Usage

### Complete Pipeline

The easiest way to run the complete pipeline is using the orchestrator script:

```bash
# Run the entire pipeline
python run_pipeline.py --all

# Or run specific stages
python run_pipeline.py --extract  # Only run model extraction
python run_pipeline.py --process  # Only run data processing
python run_pipeline.py --analyze  # Only run analysis
```

### Individual Stages

You can also run each stage separately:

#### 1. Model Extraction

Extract model architectures from Hugging Face:

```bash
cd model_extract
python run.py
```

This will:
- Download models from Hugging Face
- Extract their architectural structure
- Save the data to CSV files in `model_architecture/raw/`

#### 2. Data Processing

Process the extracted model files:

```bash
cd data_processed
python rename_csv_files.py
python deduplication.py
```

This will:
- Rename files based on their Model ID
- Deduplicate models to find unique architectures
- Save processed files to `model_architecture/processed/`

#### 3. Architecture Analysis

Analyze the processed model architectures:

```bash
cd analysis
python run.py
```

This will:
- Analyze layer types and block distributions
- Compare architectures across different tasks
- Generate visualizations and reports
- Save results to `analysis_result/`

## Analysis Outputs

The analysis produces several types of outputs:

- **Visualizations**: Bar charts and heatmaps showing layer distributions
- **CSV Data**: Comparative data between different model types
- **Reports**: Markdown reports explaining architectural patterns
- **Statistics**: Text files with summary statistics for each task

## Environment

This project requires Python 3.9+ and depends on the following main libraries:
- pandas
- numpy
- matplotlib
- seaborn
- torch
- transformers
- huggingface_hub

The complete environment is specified in `deep_nn_structure_env.yml`.

## Project Customization

The project is designed to be customizable:

- **Model Filters**: Adjust `MIN_MODELS_THRESHOLD` in `model_extract/config.py` to control which tasks are processed
- **Output Directories**: Paths can be configured in the respective config.py files
- **Visualization Options**: Modify visualization parameters in `analysis/advanced_analysis.py`

## Acknowledgements

This project uses the Hugging Face model hub and transformers library. 