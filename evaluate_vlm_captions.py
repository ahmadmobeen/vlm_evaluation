#!/usr/bin/env python3
"""
VLM Caption Evaluation System

Evaluates VLM-generated captions using GPT-4o with multi-run averaging and statistical analysis.
Compares all models against a ground truth model (default: gemini-2.5-pro) on a chunk-by-chunk basis.

Usage:
    python evaluate_vlm_captions.py --captions-dir reports_2_4_0/vss_generate_captions_results --output-dir caption_eval_results --openai-api-key YOUR_KEY
    python evaluate_vlm_captions.py --captions-dir reports_2_4_0/vss_generate_captions_results --model-to-evaluate vila-1.5 --output-dir caption_eval_results --openai-api-key YOUR_KEY
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from caption_evaluator import CaptionEvaluator, MultiRunCaptionEvaluator
from visualization_utils import CaptionVisualizationUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VLMCaptionEvaluationSystem:
    """Complete VLM caption evaluation system with multi-run averaging."""
    
    def __init__(self, captions_dir: str, output_dir: str, openai_api_key: str, 
                 ground_truth_model: str = "gemini-2.5-pro", batch_size: int = 5,
                 debug_max_chunks: Optional[int] = None, prompt_type: str = "detailed", 
                 max_concurrent_requests: int = 10, batch_mode: bool = False):
        self.captions_dir = Path(captions_dir)
        self.output_dir = Path(output_dir)
        self.openai_api_key = openai_api_key
        self.ground_truth_model = ground_truth_model
        self.batch_size = batch_size
        self.debug_max_chunks = debug_max_chunks
        self.prompt_type = prompt_type
        self.max_concurrent_requests = max_concurrent_requests
        self.batch_mode = batch_mode
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluator
        self.multi_run_evaluator = MultiRunCaptionEvaluator(openai_api_key, ground_truth_model, batch_size, prompt_type, max_concurrent_requests, batch_mode)
        
        # Initialize visualization utils
        self.viz_utils = CaptionVisualizationUtils(self.output_dir, prompt_type)
        
        # Log the configuration being used
        logger.info(f"🎯 Using {prompt_type} evaluation prompt")
        if self.batch_mode:
            logger.info(f"🚀 Concurrent batching: {self.max_concurrent_requests} concurrent batches, {self.batch_size} chunks per batch")
        else:
            logger.info(f"Individual chunk processing with concurrency limit: {self.max_concurrent_requests}")
        
    def get_available_prompt_types(self) -> Dict[str, str]:
        """Get available prompt types from the caption evaluator."""
        from caption_evaluator import CaptionPromptManager
        prompt_manager = CaptionPromptManager()
        return prompt_manager.get_available_prompts()
        
    def discover_models(self) -> List[str]:
        """Discover all available models in the captions directory."""
        models = []
        
        for item in self.captions_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                models.append(item.name)
        
        logger.info(f"📂 Discovered {len(models)} models: {models}")
        return models
    
    def discover_videos(self, model_name: str) -> List[str]:
        """Discover all videos for a specific model."""
        model_dir = self.captions_dir / model_name
        videos = []
        
        for item in model_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                videos.append(item.name)
        
        logger.info(f"🎬 Found {len(videos)} videos for model {model_name}: {videos}")
        return videos
    
    def load_caption_file(self, model_name: str, video_id: str) -> Optional[Dict[str, Any]]:
        """Load caption file for a specific model and video."""
        caption_file = self.captions_dir / model_name / video_id / f"vlm_captions_{video_id}.json"
        
        if not caption_file.exists():
            logger.warning(f"⚠️  Caption file not found: {caption_file}")
            return None
        
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"📄 Loaded {len(data.get('chunk_responses', []))} chunks for {model_name}/{video_id}")
            return data
        except Exception as e:
            logger.error(f"❌ Error loading caption file {caption_file}: {str(e)}")
            return None
    
    def validate_chunk_alignment(self, ground_truth_chunks: List[Dict], 
                                test_chunks: List[Dict], video_id: str, 
                                ground_truth_model: str, test_model: str) -> bool:
        """Validate that chunks are properly aligned between models."""
        
        if len(ground_truth_chunks) != len(test_chunks):
            logger.error(
                f"❌ Chunk count mismatch for {video_id}: "
                f"{ground_truth_model} has {len(ground_truth_chunks)} chunks, "
                f"{test_model} has {len(test_chunks)} chunks"
            )
            return False
        
        for i, (gt_chunk, test_chunk) in enumerate(zip(ground_truth_chunks, test_chunks)):
            # Convert times to float for comparison
            try:
                gt_start = float(gt_chunk['start_time'])
                gt_end = float(gt_chunk['end_time'])
                gt_chunk_id = gt_chunk.get('chunk_id', i)
                test_start = float(test_chunk['start_time'])
                test_end = float(test_chunk['end_time'])
                test_chunk_id = test_chunk.get('chunk_id', i)
                
                # Allow for small floating point differences (1ms tolerance)
                if abs(gt_start - test_start) > 0.001 or abs(gt_end - test_end) > 0.001:
                    logger.error(
                        f"❌ Chunk {gt_chunk_id} time mismatch for {video_id}: "
                        f"{ground_truth_model}: {gt_start}-{gt_end}s, "
                        f"{test_model}: {test_start}-{test_end}s"
                    )
                    return False
                if gt_chunk_id != test_chunk_id:
                    logger.error(
                        f"❌ Chunk ID mismatch for {video_id}: "
                        f"{ground_truth_model} chunk ID {gt_chunk_id}, "
                        f"{test_model} chunk ID {test_chunk_id}"
                    )
                    return False
                    
            except (ValueError, KeyError) as e:
                logger.error(f"❌ Invalid chunk time format in chunk {gt_chunk_id} for {video_id}: {str(e)}")
                return False
        
        logger.debug(f"✅ Chunk alignment validated for {video_id}: {len(ground_truth_chunks)} chunks")
        return True
    
    def prepare_evaluation_pairs(self, models_to_evaluate: List[str]) -> List[Tuple[str, str, str]]:
        """Prepare all model-video pairs for evaluation."""
        evaluation_pairs = []
        
        # Get ground truth videos
        ground_truth_videos = self.discover_videos(self.ground_truth_model)
        
        for model_name in models_to_evaluate:
            if model_name == self.ground_truth_model:
                continue
                
            model_videos = self.discover_videos(model_name)
            
            # Find common videos
            common_videos = set(ground_truth_videos) & set(model_videos)
            
            if not common_videos:
                logger.warning(f"⚠️  No common videos found between {self.ground_truth_model} and {model_name}")
                continue
            
            logger.info(f"📊 Will evaluate {len(common_videos)} common videos for {model_name}")
            
            for video_id in common_videos:
                evaluation_pairs.append((model_name, video_id, self.ground_truth_model))
        
        return evaluation_pairs
    
    def organize_chunks_for_evaluation(self, evaluation_pairs: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """Organize all chunks for evaluation with validation."""
        
        organized_data = {
            "ground_truth_model": self.ground_truth_model,
            "evaluation_pairs": [],
            "chunk_evaluations": []
        }
        
        for model_name, video_id, gt_model in evaluation_pairs:
            logger.info(f"📋 Processing {model_name}/{video_id} vs {gt_model}")
            
            # Load caption files
            gt_data = self.load_caption_file(gt_model, video_id)
            test_data = self.load_caption_file(model_name, video_id)
            
            if not gt_data or not test_data:
                logger.warning(f"⚠️  Skipping {model_name}/{video_id} - missing caption data")
                continue
            
            gt_chunks = gt_data.get("chunk_responses", [])
            test_chunks = test_data.get("chunk_responses", [])

            if self.debug_max_chunks:
                gt_chunks = gt_chunks[:self.debug_max_chunks]
                test_chunks = test_chunks[:self.debug_max_chunks]
                logger.warning(f"🐛 DEBUG MODE: Limiting to first {self.debug_max_chunks} chunks")
            
            # Validate chunk alignment
            if not self.validate_chunk_alignment(gt_chunks, test_chunks, video_id, gt_model, model_name):
                logger.warning(f"⚠️  Skipping {model_name}/{video_id} - chunk alignment failed")
                continue
            
            # Add to organized data
            organized_data["evaluation_pairs"].append({
                "model_name": model_name,
                "video_id": video_id,
                "chunk_count": len(gt_chunks)
            })
            
            # Create chunk-by-chunk evaluation tasks
            for i, (gt_chunk, test_chunk) in enumerate(zip(gt_chunks, test_chunks)):
                chunk_eval = {
                    "model_name": model_name,
                    "video_id": video_id,
                    "chunk_index": i,
                    "chunk_id": gt_chunk.get("chunk_id", i),
                    "start_time": gt_chunk["start_time"],
                    "end_time": gt_chunk["end_time"],
                    "ground_truth_content": gt_chunk["content"],
                    "test_content": test_chunk["content"],
                    "evaluation_id": f"{model_name}_{video_id}_chunk_{i}"
                }
                organized_data["chunk_evaluations"].append(chunk_eval)
        
        # Apply debug mode chunk limiting if specified
        # if self.debug_max_chunks and len(organized_data["chunk_evaluations"]) > self.debug_max_chunks:
        #     logger.warning(f"🐛 DEBUG MODE: Limiting chunks to {self.debug_max_chunks} (from {len(organized_data['chunk_evaluations'])})")
        #     organized_data["chunk_evaluations"] = organized_data["chunk_evaluations"][:self.debug_max_chunks]
            
        #     # Also update evaluation_pairs to reflect the limited chunks
        #     processed_pairs = set()
        #     limited_pairs = []
        #     for chunk in organized_data["chunk_evaluations"]:
        #         pair_key = (chunk["model_name"], chunk["video_id"])
        #         if pair_key not in processed_pairs:
        #             processed_pairs.add(pair_key)
        #             # Count chunks for this pair
        #             pair_chunks = [c for c in organized_data["chunk_evaluations"] 
        #                          if c["model_name"] == chunk["model_name"] and c["video_id"] == chunk["video_id"]]
        #             limited_pairs.append({
        #                 "model_name": chunk["model_name"],
        #                 "video_id": chunk["video_id"],
        #                 "chunk_count": len(pair_chunks)
        #             })
        #     organized_data["evaluation_pairs"] = limited_pairs
        
        logger.info(f"📊 Organized {len(organized_data['chunk_evaluations'])} chunk evaluations")
        logger.info(f"📈 Models to evaluate: {len(set(ep['model_name'] for ep in organized_data['evaluation_pairs']))}")
        
        return organized_data
    
    def create_summary_statistics(self, evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create summary statistics table from evaluation results."""
        
        # Group evaluations by model
        model_groups = {}
        for eval_result in evaluations:
            if "error" in eval_result:
                continue
                
            model_name = eval_result["model_name"]
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(eval_result)
        
        # Calculate statistics
        criteria = ["factual_accuracy", "completeness", "clarity", "conciseness", "temporal_accuracy"]
        summary_data = []
        
        for model_name, evals in model_groups.items():
            model_stats = {"model": model_name}
            
            for criterion in criteria:
                scores = [e[criterion]["score"] for e in evals if criterion in e]
                if scores:
                    model_stats[f"{criterion}_mean"] = np.mean(scores)
                    model_stats[f"{criterion}_std"] = np.std(scores)
                else:
                    model_stats[f"{criterion}_mean"] = 0
                    model_stats[f"{criterion}_std"] = 0
            
            # Overall score
            overall_scores = [e["overall_score"] for e in evals if "overall_score" in e]
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
    
    def save_detailed_results(self, organized_data: Dict[str, Any], 
                            multi_run_results: Dict[str, Any], 
                            summary_stats: pd.DataFrame):
        """Save all detailed evaluation results."""
        
        # Save organized evaluation data
        organized_file = self.output_dir / "organized_evaluation_data.json"
        with open(organized_file, 'w', encoding='utf-8') as f:
            json.dump(organized_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved organized data: {organized_file}")
        
        # Save multi-run results
        multi_run_file = self.output_dir / "multi_run_caption_evaluations.json"
        with open(multi_run_file, 'w', encoding='utf-8') as f:
            json.dump(multi_run_results, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved multi-run results: {multi_run_file}")
        
        # Save averaged evaluations
        averaged_file = self.output_dir / "averaged_caption_evaluations.json"
        with open(averaged_file, 'w', encoding='utf-8') as f:
            json.dump(multi_run_results["averaged_results"], f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved averaged results: {averaged_file}")
        
        # Save summary statistics
        summary_csv = self.output_dir / "caption_evaluation_summary.csv"
        summary_stats.to_csv(summary_csv, index=False)
        logger.info(f"💾 Saved summary statistics: {summary_csv}")
        
        # Save formatted summary table
        formatted_table = self.create_formatted_summary_table(summary_stats)
        table_file = self.output_dir / "formatted_summary_table.txt"
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write(formatted_table)
        logger.info(f"💾 Saved formatted table: {table_file}")
    
    def create_formatted_summary_table(self, summary_stats: pd.DataFrame) -> str:
        """Create formatted markdown-style summary table."""
        
        if summary_stats.empty:
            return "No evaluation results to display."
        
        table_lines = []
        table_lines.append("# VLM Caption Evaluation Results")
        table_lines.append("")
        table_lines.append("| Model | Factual Accuracy | Completeness | Clarity | Conciseness | Temporal Accuracy | Overall |")
        table_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        
        for _, row in summary_stats.iterrows():
            model = row["model"]
            fa = f"{row['factual_accuracy_mean']:.1f} ± {row['factual_accuracy_std']:.2f}"
            comp = f"{row['completeness_mean']:.1f} ± {row['completeness_std']:.2f}"
            clar = f"{row['clarity_mean']:.1f} ± {row['clarity_std']:.2f}"
            conc = f"{row['conciseness_mean']:.1f} ± {row['conciseness_std']:.2f}"
            temp = f"{row['temporal_accuracy_mean']:.1f} ± {row['temporal_accuracy_std']:.2f}"
            overall = f"{row['overall_mean']:.1f} ± {row['overall_std']:.2f}"
            
            table_lines.append(f"| `{model}` | {fa} | {comp} | {clar} | {conc} | {temp} | {overall} |")
        
        return "\n".join(table_lines)
    
    async def run_evaluation(self, models_to_evaluate: Optional[List[str]] = None, 
                           num_runs: int = 3) -> Dict[str, Any]:
        """Run complete evaluation with multi-run averaging."""
        
        logger.info("🚀 Starting VLM Caption Evaluation System")
        logger.info(f"📂 Captions directory: {self.captions_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎯 Ground truth model: {self.ground_truth_model}")
        logger.info(f"🔄 Number of runs: {num_runs}")
        
        # Discover models
        available_models = self.discover_models()
        
        if not models_to_evaluate:
            models_to_evaluate = available_models
        else:
            # Validate specified models
            missing_models = set(models_to_evaluate) - set(available_models)
            if missing_models:
                logger.error(f"❌ Specified models not found: {missing_models}")
                return {"success": False, "error": f"Missing models: {missing_models}"}
        
        # Remove ground truth model from evaluation list
        models_to_evaluate = [m for m in models_to_evaluate if m != self.ground_truth_model]
        
        logger.info(f"📊 Models to evaluate: {models_to_evaluate}")
        
        if not models_to_evaluate:
            logger.error("❌ No models to evaluate after filtering")
            return {"success": False, "error": "No models to evaluate"}
        
        # Prepare evaluation pairs
        evaluation_pairs = self.prepare_evaluation_pairs(models_to_evaluate)
        
        if not evaluation_pairs:
            logger.error("❌ No valid evaluation pairs found")
            return {"success": False, "error": "No valid evaluation pairs"}
        
        # Organize chunks for evaluation
        organized_data = self.organize_chunks_for_evaluation(evaluation_pairs)
        
        if not organized_data["chunk_evaluations"]:
            logger.error("❌ No valid chunk evaluations prepared")
            return {"success": False, "error": "No valid chunk evaluations"}
        
        # Run multi-run evaluation
        logger.info(f"🤖 Running {num_runs}-run GPT-4o evaluation...")
        multi_run_results = await self.multi_run_evaluator.run_multiple_evaluations(
            organized_data["chunk_evaluations"], num_runs
        )
        
        # Create summary statistics
        averaged_evaluations = multi_run_results["averaged_results"]
        summary_stats = self.create_summary_statistics(averaged_evaluations)
        
        # Save all results
        self.save_detailed_results(organized_data, multi_run_results, summary_stats)
        
        # Generate visualizations
        logger.info("📊 Generating visualizations...")
        self.viz_utils.create_all_visualizations(summary_stats, averaged_evaluations)
        
        # Print summary
        self.print_evaluation_summary(summary_stats, multi_run_results, organized_data)
        
        return {
            "success": True,
            "summary_stats": summary_stats,
            "multi_run_results": multi_run_results,
            "organized_data": organized_data
        }
    
    def print_evaluation_summary(self, summary_stats: pd.DataFrame, 
                               multi_run_results: Dict[str, Any], 
                               organized_data: Dict[str, Any]):
        """Print evaluation summary to console."""
        
        logger.info("\n" + "="*80)
        logger.info("🎯 VLM CAPTION EVALUATION COMPLETE")
        logger.info("="*80)
        
        # Safe access to organized_data fields
        total_videos = 0
        total_chunks = 0
        
        if 'evaluation_pairs' in organized_data:
            total_videos = len(set(ep['video_id'] for ep in organized_data['evaluation_pairs']))
        elif 'chunk_evaluations' in organized_data:
            # Fallback: count unique video_ids from chunk evaluations
            total_videos = len(set(ce.get('video_id', 'unknown') for ce in organized_data['chunk_evaluations']))
        
        if 'chunk_evaluations' in organized_data:
            total_chunks = len(organized_data['chunk_evaluations'])
        
        logger.info(f"🎬 Total videos evaluated: {total_videos}")
        logger.info(f"📊 Total chunk evaluations: {total_chunks}")
        logger.info(f"🔄 Evaluation runs: {multi_run_results.get('num_runs', 'N/A')}")
        logger.info(f"🎯 Ground truth model: {self.ground_truth_model}")
        
        if not summary_stats.empty:
            logger.info(f"\n🏆 MODEL RANKING:")
            for _, row in summary_stats.iterrows():
                logger.info(
                    f"  {int(row['rank'])}. {row['model']}: "
                    f"{row['overall_mean']:.1f} ± {row['overall_std']:.2f} "
                    f"({int(row['num_evaluations'])} evaluations)"
                )
        
        logger.info(f"\n📁 All results saved to: {self.output_dir}")
        logger.info(f"📊 Visualizations saved to: {self.output_dir}/*.png")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM captions using GPT-4o with multi-run averaging"
    )
    parser.add_argument(
        "--captions-dir",
        required=True,
        help="Directory containing VLM caption results"
    )
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--ground-truth-model",
        default="gemini-2.5-pro",
        help="Ground truth model name (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--model-to-evaluate",
        help="Specific model to evaluate (default: evaluate all models)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of evaluation runs for averaging (default: 3)"
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Enable batch processing mode (default: False)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of chunks to evaluate in each API call (default: 5)"
    )
    parser.add_argument(
        "--max-concurrent-batches",
        type=int,
        default=10,
        help="Maximum number of concurrent batch requests (default: 10)"
    )
    parser.add_argument(
        "--debug-max-chunks",
        type=int,
        help="Debug mode: limit total chunks processed (useful for quick testing)"
    )
    parser.add_argument(
        "--prompt-type",
        default="detailed",
        choices=["detailed", "coverage_hallucination", "completeness"],
        help="Evaluation prompt type (default: detailed)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OpenAI API key required. Set --openai-api-key or OPENAI_API_KEY env var")
        return 1
    
    # Validate directories
    if not Path(args.captions_dir).exists():
        logger.error(f"❌ Captions directory not found: {args.captions_dir}")
        return 1
    
    # Initialize evaluation system
    eval_system = VLMCaptionEvaluationSystem(
        args.captions_dir,
        args.output_dir,
        api_key,
        args.ground_truth_model,
        args.batch_size,
        args.debug_max_chunks,
        args.prompt_type,
        args.max_concurrent_batches,
        args.batch_mode
    )
    
    # Prepare models list
    models_to_evaluate = [args.model_to_evaluate] if args.model_to_evaluate else None
    
    # Run evaluation
    try:
        result = asyncio.run(eval_system.run_evaluation(models_to_evaluate, args.num_runs))
        return 0 if result["success"] else 1
    except KeyboardInterrupt:
        logger.warning("⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    sys.exit(main())