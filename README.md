# VLM Caption Evaluation

This repository provides tools to evaluate Visual Language Model (VLM) generated captions using GPT-4o as an evaluator, along with visualization and data export utilities.

## Scripts Overview

### 1. Main Evaluation Script
- **`evaluate_vlm_captions.py`** - Runs GPT-4o evaluation on VLM captions with multi-run averaging

### 2. Visualization Scripts
- **`generate_visualizations_from_results.py`** - Creates visualizations from existing evaluation results
- **`visualization_utils.py`** - Utility functions for creating various plots (radar, bar, heatmap, violin, box plots, etc.)

### 3. Data Export Scripts
- **`generate_csv_from_data.py`** - Exports VLM captions and GPT-4o evaluations to CSV format

## Usage

### Running Full Evaluation

To reproduce the results found in the `reports` directory, run the following command:

```bash
python evaluate_vlm_captions.py \
  --captions-dir reports/minseok_prompt_audio_x_caption_sorted \
  --num-runs 3 \
  --output-dir reports/caption_full_comparison_minseok_audio_x_caption_sorted_completeness \
  --prompt-type completeness \
  --openai-api-key YOUR_OPENAI_API_KEY
```

**Parameters:**
- `--captions-dir`: Path to the directory containing VLM-generated captions
- `--num-runs`: Number of evaluation runs to perform for averaging (default: 3)
- `--output-dir`: Directory to save the evaluation results
- `--prompt-type`: Type of evaluation prompt (`detailed`, `completeness`, or `coverage_hallucination`)
- `--openai-api-key`: OpenAI API key (or set `OPENAI_API_KEY` environment variable)
- `--batch-mode`: Enable batch processing for faster evaluation
- `--max-concurrent-batches`: Maximum concurrent API requests (default: 10)

### Generating Visualizations from Existing Results

To create visualizations without re-running evaluations:

```bash
python generate_visualizations_from_results.py \
  --results-dir reports/caption_full_comparison_minseok_audio_x_caption_sorted_completeness \
  --output-dir visualization_output
```

**Parameters:**
- `--results-dir`: Directory containing existing evaluation results
- `--output-dir`: Output directory for visualization files
- `--prompt-type`: Evaluation prompt type (auto-detected if not specified)
- `--visualization-type`: Specific visualization to generate (`all`, `radar`, `bar`, `heatmap`, `consistency`, `violin`, `box`, `temporal`)
- `--summary-only`: Print data summary without generating visualizations

**Available Visualizations:**
- **Radar Plot**: Multi-criteria comparison across models
- **Bar Chart**: Overall score comparison with error bars
- **Heatmap**: Criteria scores matrix for all models
- **Consistency Analysis**: Standard deviation analysis
- **Violin Plots**: Score distribution shapes
- **Box Plots**: Quartiles, median, mean, and outliers *(New!)*
- **Temporal Analysis**: Specialized temporal accuracy analysis

### Exporting Data to CSV

To export VLM captions and GPT-4o evaluations to CSV format:

```bash
python generate_csv_from_data.py \
  --captions-dir reports/minseok_prompt_audio_x_caption_sorted \
  --evaluations-dir reports/caption_full_comparison_minseok_audio_x_caption_sorted_completeness \
  --output-dir csv_output
```

**Parameters:**
- `--captions-dir`: Directory containing VLM caption data
- `--evaluations-dir`: Directory containing GPT-4o evaluation results
- `--output-dir`: Output directory for CSV files
- `--csv-type`: Type of CSV to generate (`both`, `captions`, `evaluations`)
- `--summary-only`: Print data summary without generating CSV files

**Generated CSV Files:**
- **`vlm_captions.csv`**: Contains actual captions from all models
  - Columns: `video_id`, `chunk_id`, `start_time`, `end_time`, `vila-1.5`, `nvila`, `cosmos_reason1`, `qwen3-vl-30b-a3b-instruct`, `gemini-2.5-pro`
- **`gpt4o_evaluations.csv`**: Contains GPT-4o judgments and scores
  - Columns: `video_id`, `chunk_id`, `start_time`, `end_time`, `[model]`, `[model]_score` for each evaluated model

## Output Files

The evaluation process generates several output files:

### Evaluation Results
- `averaged_caption_evaluations.json` - Averaged results across multiple runs
- `multi_run_caption_evaluations.json` - Individual results from each run
- `caption_evaluation_summary.csv` - Summary statistics table
- `organized_evaluation_data.json` - Raw organized evaluation data
- `formatted_summary_table.txt` - Human-readable summary table

### Visualizations
- `radar_plot_comparison.png` - Multi-criteria radar plot
- `bar_chart_overall_scores.png` - Overall score comparison
- `heatmap_criteria_scores.png` - Criteria scores heatmap
- `consistency_analysis.png` - Model consistency analysis
- `score_distributions.png` - Violin plots of score distributions
- `box_plot_distributions.png` - Box plots with quartiles and outliers *(New!)*
- `temporal_accuracy_analysis.png` - Temporal accuracy specialized analysis

## Evaluation Criteria

### Detailed Prompt
For `prompt-type: detailed`, evaluations include scores (1-10) for:
- **Factual Accuracy**: Correctness of described objects, actions, and events
- **Completeness**: Coverage of important visual elements from ground truth
- **Clarity**: How clear, well-structured, and understandable the caption is
- **Conciseness**: Balance between detail and brevity
- **Temporal Accuracy**: Correct handling of timestamps and temporal sequences

### Other Prompt Types
For `completeness` and `coverage_hallucination` prompts:
- Single overall score reflecting caption quality
- Detailed criteria scores will be 0 in reports
- Focus on specific aspects (completeness vs. hallucination detection)

## Requirements

Install dependencies:
```bash
pip install matplotlib numpy pandas seaborn openai
```

Or using the project configuration:
```bash
pip install -e .
```
