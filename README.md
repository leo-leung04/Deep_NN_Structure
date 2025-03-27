# Hugging Face Model Architecture Extractor

This project is used to extract model architectures from Hugging Face for various tasks and organize them by category. Each base model will only be extracted once to avoid duplication.

## Features

- Retrieves all task categories from Hugging Face
- Extracts models for each task category
- Extracts detailed layer-level structure (blocks and layers) for each model
- Generates architecture information CSV files for each base model
- Adds model metadata as the last row in the CSV file
- Also saves complete metadata in JSON format
- Organizes file structure by task and category

## Project Structure

```
├── main.py                # Main entry file
├── model_utils.py         # Model utilities (create models and extract architectures)
├── api_utils.py           # API utilities (retrieve model metadata)
├── task_utils.py          # Task utilities (retrieve task categories)
├── config.py              # Configuration file (logging and predefined task lists)
├── logs/                  # Log output directory
│   └── model_extraction.log
└── huggingface_models/    # Output directory (organized by category/task)
    ├── Computer Vision/
    │   ├── Image Classification/
    │   │   └── *_structure.csv
    │   └── ...
    ├── Natural Language Processing/
    │   ├── Text Generation/
    │   │   └── *_structure.csv
    │   └── ...
    └── ...
```

## Installation Dependencies

```bash
pip install huggingface-hub torch transformers pandas tqdm beautifulsoup4 requests
```

## Usage

1. Run the main script directly:

```bash
python main.py
```

2. View output results:
   - Model architecture information is saved in the `huggingface_models` directory
   - Each model generates a `{base_model_name}_structure.csv` file
   - Metadata is also saved as `{base_model_name}_metadata.json`

## Output Format

### CSV File Structure

Each CSV file contains the detailed layer structure of the model, including:
- Layer Name
- Layer Type
- Input Shape
- Output Shape
- Parameters

At the last row of the CSV file, a special `METADATA` row is added, containing all model metadata, such as:
- Model ID
- Base Model Name
- Task and Category
- Author
- Download Count
- Etc.

### Example Structure

```
Layer Name,Layer Type,Input Shape,Output Shape,Parameters,Model ID,Base Model Name,Task,Category
bert.embeddings,BertEmbeddings,[1, 10],[1, 10, 768],{},,,,,
bert.encoder,BertEncoder,[1, 10, 768],[1, 10, 768],{},,,,,
bert.pooler,BertPooler,[1, 10, 768],[1, 768],{},,,,,
...
METADATA,,,,,"bert-base-uncased","bert","Text Classification","Natural Language Processing"
```

## Notes

- Due to the need to download and load models, this tool requires a good network connection and sufficient hardware resources
- For each task, a maximum of 100 models are processed by default
- You can adjust the `max_models_per_task` parameter in `main.py` to change the number of models processed 