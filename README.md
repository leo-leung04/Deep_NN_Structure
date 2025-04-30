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
├── model_extract/                 # Model extraction scripts
│   ├── config.py                  # Configuration settings
│   ├── utils.py                   # Utility functions
│   ├── model_loader.py            # Model loading functions
│   ├── structure_extractor.py     # Structure extraction functions
│   ├── main.py                    # Main extraction logic
│   └── run.py                     # Entry point script
├── model_architecture/            # Extracted model architectures
│   ├── raw/                       # Raw extracted model files
│   ├── process_with_name/         # Name-based deduplicated files
│   └── process_with_architecture/ # Architecture-based deduplicated files
├── data_processing/               # Data processing scripts
│   ├── __init__.py                # Package initialization
│   ├── rename_csv_files.py        # File renaming script
│   ├── deduplicate_with_name.py   # Name-based deduplication
│   ├── deduplicate_with_architecture.py # Architecture-based deduplication
│   └── deduplication_runner.py    # Runner for selecting deduplication strategy
├── model_architecture_analyzer.py # Main analysis script
├── analysis_result/               # Analysis results (default output directory)
│   ├── images/                    # Visualizations and plots
│   │   ├── boxplot/               # Layer distribution boxplots by task
│   │   └── ...                    # Other image files and plots
│   ├── csv_data/                  # Processed CSV data
│   ├── reports/                   # Analysis reports
│   └── statistics_txt/            # Statistical summaries
├── .github/workflows/             # GitHub Actions workflows
│   └── model_extraction.yml       # Pipeline workflow definition
├── run_pipeline.py                # Full pipeline orchestrator
└── deep_nn_structure_env.yml      # Environment file
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
# Run the entire pipeline with default settings (name-based deduplication)
python run_pipeline.py --all

# Run with architecture-based deduplication
python run_pipeline.py --all --dedup-strategy architecture --similarity-threshold 0.85

# Run with both deduplication strategies
python run_pipeline.py --all --dedup-strategy both

# Or run specific stages
python run_pipeline.py --extract  # Only run model extraction
python run_pipeline.py --process --dedup-strategy architecture  # Only run data processing with architecture dedup
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
cd data_processing
python rename_csv_files.py

# Choose a deduplication strategy:
# 1. Name-based deduplication:
python deduplication_runner.py --strategy name

# 2. Architecture-based deduplication:
python deduplication_runner.py --strategy architecture --similarity-threshold 0.85

# 3. Both strategies:
python deduplication_runner.py --strategy both
```

This will:
- Rename files based on their Model ID
- Deduplicate models using the selected strategy:
  - Name-based: Groups models by base name (e.g., "bert", "gpt")
  - Architecture-based: Groups models by structural similarity
- Save processed files to appropriate directories:
  - Name-based: `model_architecture/process_with_name/`
  - Architecture-based: `model_architecture/process_with_architecture/`

#### 3. Architecture Analysis

Analyze the processed model architectures:

```bash
# Analyze name-based deduplicated results
python model_architecture_analyzer.py --input_dir="model_architecture/process_with_name"
# Output will be automatically saved to analysis_result_process_with_name directory

# Or analyze architecture-based deduplicated results
python model_architecture_analyzer.py --input_dir="model_architecture/process_with_architecture"
# Output will be automatically saved to analysis_result_process_with_architecture directory

# You can also specify a custom output directory
python model_architecture_analyzer.py --input_dir="model_architecture/process_with_name" --output_dir="analysis_result_name_based"

# To preserve all analysis versions, add a timestamp
python model_architecture_analyzer.py --input_dir="model_architecture/process_with_name" --append_timestamp
```

This will:
- Analyze layer types and block distributions
- Compare architectures across different tasks
- Generate various visualizations and reports:
  - Layer type distribution charts
  - Heatmaps (both original and log-scaled)
  - Hierarchy charts
  - Nesting level analysis charts
  - Layer distribution boxplots for each task
- Save all results to the output directory

## Analysis Outputs

The analysis produces several types of outputs:

- **Visualizations**:
  - **Distribution Charts**: Bar charts showing layer type distributions
  - **Heatmaps**: Original and log-scaled heatmaps of layer distributions
  - **Hierarchy Charts**: Visualizations of model hierarchical structures
  - **Nesting Level Charts**: Mean and median nesting level analysis
  - **Boxplots**: Distribution of top 10 layer types across models for each task
  
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

- **GitHub Variables**: Configure the pipeline using repository variables without modifying code
- **Model Filters**: Adjust `MIN_MODELS_THRESHOLD` in `model_extract/config.py` or set the `MODEL_EXTRACT_MIN_MODELS` variable
- **Output Directories**: Paths can be configured in the respective config files or via environment variables
- **Deduplication Settings**: Adjust similarity threshold for architecture-based deduplication
- **Visualization Options**: Modify visualization parameters in `model_architecture_analyzer.py`

## Acknowledgements

This project uses the Hugging Face model hub and transformers library. 