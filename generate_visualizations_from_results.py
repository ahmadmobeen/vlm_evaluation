#!/usr/bin/env python3
"""
Generate Visualizations from Existing VLM Caption Evaluations

This script loads existing evaluation data and generates visualizations without
running new GPT-4o evaluations. Useful for re-generating plots or creating
visualizations with different settings.

Usage:
    python generate_visualizations_from_results.py --results-dir reports/caption_full_comparison_minseok_audio_x_caption_sorted_completeness --output-dir new_viz_output
    python generate_visualizations_from_results.py --results-dir reports/caption_full_comparison_minseok_audio_x_caption_sorted_completeness --output-dir new_viz_output --prompt-type completeness
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from visualization_utils import CaptionVisualizationUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Generate visualizations from existing evaluation data."""
    
    def __init__(self, results_dir: str, output_dir: str, prompt_type: Optional[str] = None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect prompt type if not provided
        self.prompt_type = prompt_type or self.detect_prompt_type()
        
        # Initialize visualization utils
        self.viz_utils = CaptionVisualizationUtils(self.output_dir, self.prompt_type)
        
        logger.info(f"📊 Detected/Using prompt type: {self.prompt_type}")
    
    def detect_prompt_type(self) -> str:
        """Auto-detect the prompt type from the evaluation data."""
        
        # Try to load a sample evaluation to detect the format
        averaged_file = self.results_dir / "averaged_caption_evaluations.json"
        
        if averaged_file.exists():
            try:
                with open(averaged_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data and len(data) > 0:
                    sample_eval = data[0]
                    
                    # Check for detailed format (has specific criteria)
                    if "factual_accuracy" in sample_eval:
                        return "detailed"
                    # Check for completeness format (has score/judgment but no detailed criteria)
                    elif "score" in sample_eval and "judgment" in sample_eval:
                        return "completeness"
                    # Check for coverage_hallucination format
                    elif "score" in sample_eval:
                        return "coverage_hallucination"
                
            except Exception as e:
                logger.warning(f"⚠️  Could not auto-detect prompt type: {e}")
        
        # Default fallback
        logger.info("🔍 Using default prompt type: detailed")
        return "detailed"
    
    def load_evaluation_data(self) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
        """Load evaluation data and summary statistics."""
        
        # Load averaged evaluations
        averaged_file = self.results_dir / "averaged_caption_evaluations.json"
        if not averaged_file.exists():
            raise FileNotFoundError(f"Averaged evaluations file not found: {averaged_file}")
        
        with open(averaged_file, 'r', encoding='utf-8') as f:
            evaluations = json.load(f)
        
        logger.info(f"📄 Loaded {len(evaluations)} averaged evaluations")
        
        # Load summary statistics
        summary_file = self.results_dir / "caption_evaluation_summary.csv"
        if summary_file.exists():
            summary_stats = pd.read_csv(summary_file)
            logger.info(f"📊 Loaded summary statistics for {len(summary_stats)} models")
        else:
            # Generate summary statistics if CSV doesn't exist
            logger.info("📊 Generating summary statistics from evaluation data")
            summary_stats = self.create_summary_statistics(evaluations)
        
        return evaluations, summary_stats
    
    def create_summary_statistics(self, evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create summary statistics from evaluation data."""
        
        # Group evaluations by model
        model_groups = {}
        for eval_result in evaluations:
            if "error" in eval_result:
                continue
                
            model_name = eval_result["model_name"]
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(eval_result)
        
        # Calculate statistics based on prompt type
        summary_data = []
        
        for model_name, evals in model_groups.items():
            model_stats = {"model": model_name}
            
            if self.prompt_type == "detailed":
                # For detailed evaluations, we have specific criteria
                criteria = ["factual_accuracy", "completeness", "clarity", "conciseness", "temporal_accuracy"]
                
                for criterion in criteria:
                    scores = []
                    for e in evals:
                        if criterion in e and isinstance(e[criterion], dict) and "score" in e[criterion]:
                            scores.append(e[criterion]["score"])
                        elif f"{criterion}_score" in e:  # Alternative format
                            scores.append(e[f"{criterion}_score"])
                    
                    if scores:
                        model_stats[f"{criterion}_mean"] = np.mean(scores)
                        model_stats[f"{criterion}_std"] = np.std(scores)
                    else:
                        model_stats[f"{criterion}_mean"] = 0
                        model_stats[f"{criterion}_std"] = 0
            
            else:
                # For other prompt types, use zeros for criteria we don't have
                criteria = ["factual_accuracy", "completeness", "clarity", "conciseness", "temporal_accuracy"]
                for criterion in criteria:
                    model_stats[f"{criterion}_mean"] = 0
                    model_stats[f"{criterion}_std"] = 0
            
            # Overall score (available in all formats)
            overall_scores = []
            for e in evals:
                if "overall_score" in e:
                    overall_scores.append(e["overall_score"])
                elif "score" in e:  # Fallback to 'score' field
                    overall_scores.append(e["score"])
            
            if overall_scores:
                model_stats["overall_mean"] = np.mean(overall_scores)
                model_stats["overall_std"] = np.std(overall_scores)
            else:
                model_stats["overall_mean"] = 0
                model_stats["overall_std"] = 0
            
            model_stats["num_evaluations"] = len(evals)
            summary_data.append(model_stats)
        
        # Convert to DataFrame and sort by overall score
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.sort_values("overall_mean", ascending=False)
            df["rank"] = range(1, len(df) + 1)
        
        return df
    
    def normalize_evaluations_for_visualization(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize evaluation data to work with visualization functions."""
        
        normalized = []
        
        for eval_data in evaluations:
            if "error" in eval_data:
                continue
            
            normalized_eval = eval_data.copy()
            
            # Ensure overall_score is available
            if "overall_score" not in normalized_eval and "score" in normalized_eval:
                normalized_eval["overall_score"] = normalized_eval["score"]
            
            # For non-detailed prompt types, create mock detailed criteria for visualization compatibility
            if self.prompt_type != "detailed":
                # Use the overall score for all criteria to enable visualization
                score = normalized_eval.get("overall_score", 0)
                normalized_eval["factual_accuracy"] = {"score": score}
                normalized_eval["completeness"] = {"score": score}
                normalized_eval["clarity"] = {"score": score}
                normalized_eval["conciseness"] = {"score": score}
                normalized_eval["temporal_accuracy"] = {"score": score}
            
            normalized.append(normalized_eval)
        
        return normalized
    
    def generate_all_visualizations(self) -> List[str]:
        """Generate all visualizations from existing data."""
        
        logger.info("🎨 Loading evaluation data...")
        
        # Load data
        evaluations, summary_stats = self.load_evaluation_data()
        
        # Normalize evaluations for visualization
        normalized_evaluations = self.normalize_evaluations_for_visualization(evaluations)
        
        logger.info("🎨 Generating visualizations...")
        
        # Generate all visualizations
        saved_files = self.viz_utils.create_all_visualizations(summary_stats, normalized_evaluations)
        
        logger.info(f"✅ Generated {len(saved_files)} visualization files")
        
        return saved_files
    
    def generate_specific_visualization(self, viz_type: str) -> str:
        """Generate a specific type of visualization."""
        
        logger.info(f"🎨 Loading evaluation data for {viz_type} visualization...")
        
        # Load data
        evaluations, summary_stats = self.load_evaluation_data()
        
        # Normalize evaluations for visualization
        normalized_evaluations = self.normalize_evaluations_for_visualization(evaluations)
        
        logger.info(f"🎨 Generating {viz_type} visualization...")
        
        # Generate specific visualization
        viz_functions = {
            "radar": self.viz_utils.create_radar_plot,
            "bar": self.viz_utils.create_bar_chart_comparison,
            "heatmap": self.viz_utils.create_criteria_heatmap,
            "consistency": self.viz_utils.create_std_deviation_plot,
            "violin": self.viz_utils.create_evaluation_distribution_plot,
            "box": self.viz_utils.create_box_plot_distributions,
            "temporal": self.viz_utils.create_temporal_accuracy_analysis
        }
        
        if viz_type not in viz_functions:
            raise ValueError(f"Unknown visualization type: {viz_type}. Available: {list(viz_functions.keys())}")
        
        if viz_type in ["radar", "bar", "heatmap", "consistency"]:
            # These functions use summary stats
            saved_file = viz_functions[viz_type](summary_stats)
        else:
            # These functions use raw evaluations
            saved_file = viz_functions[viz_type](normalized_evaluations)
        
        logger.info(f"✅ Generated {viz_type} visualization: {saved_file}")
        
        return saved_file
    
    def print_data_summary(self):
        """Print summary of the loaded data."""
        
        evaluations, summary_stats = self.load_evaluation_data()
        
        logger.info("\n" + "="*60)
        logger.info("📊 EVALUATION DATA SUMMARY")
        logger.info("="*60)
        
        logger.info(f"📂 Results directory: {self.results_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎯 Prompt type: {self.prompt_type}")
        
        # Count models and videos
        models = set(e["model_name"] for e in evaluations if "error" not in e)
        videos = set(e["video_id"] for e in evaluations if "error" not in e)
        
        logger.info(f"🤖 Models evaluated: {len(models)}")
        logger.info(f"🎬 Videos evaluated: {len(videos)}")
        logger.info(f"📊 Total evaluations: {len(evaluations)}")
        
        if not summary_stats.empty:
            logger.info(f"\n🏆 MODEL RANKING:")
            for _, row in summary_stats.iterrows():
                logger.info(
                    f"  {int(row['rank'])}. {row['model']}: "
                    f"{row['overall_mean']:.2f} ± {row['overall_std']:.3f} "
                    f"({int(row['num_evaluations'])} evaluations)"
                )
        
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from existing VLM caption evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing existing evaluation results"
    )
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--prompt-type",
        choices=["detailed", "coverage_hallucination", "completeness"],
        help="Evaluation prompt type (auto-detected if not specified)"
    )
    parser.add_argument(
        "--visualization-type",
        choices=["all", "radar", "bar", "heatmap", "consistency", "violin", "box", "temporal"],
        default="all",
        help="Type of visualization to generate (default: all)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print data summary, don't generate visualizations"
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not Path(args.results_dir).exists():
        logger.error(f"❌ Results directory not found: {args.results_dir}")
        return 1
    
    # Check for required files
    results_path = Path(args.results_dir)
    averaged_file = results_path / "averaged_caption_evaluations.json"
    
    if not averaged_file.exists():
        logger.error(f"❌ Required file not found: {averaged_file}")
        logger.error("This script requires 'averaged_caption_evaluations.json' from a completed evaluation.")
        return 1
    
    try:
        # Initialize visualization generator
        viz_gen = VisualizationGenerator(args.results_dir, args.output_dir, args.prompt_type)
        
        # Print data summary
        viz_gen.print_data_summary()
        
        if args.summary_only:
            logger.info("📊 Summary complete (no visualizations generated)")
            return 0
        
        # Generate visualizations
        if args.visualization_type == "all":
            saved_files = viz_gen.generate_all_visualizations()
            logger.info(f"\n🎨 Successfully generated {len(saved_files)} visualization files:")
            for file_path in saved_files:
                if file_path:  # Filter out empty results
                    logger.info(f"  📊 {Path(file_path).name}")
        else:
            saved_file = viz_gen.generate_specific_visualization(args.visualization_type)
            if saved_file:
                logger.info(f"\n🎨 Successfully generated visualization: {Path(saved_file).name}")
        
        logger.info(f"\n📁 All visualizations saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Visualization generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    sys.exit(main())