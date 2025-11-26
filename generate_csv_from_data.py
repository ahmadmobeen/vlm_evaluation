#!/usr/bin/env python3
"""
Generate CSV Files from VLM Caption Data and Evaluations

This script creates two CSV files:
1. vlm_captions.csv - Contains actual captions from all models
2. gpt4o_evaluations.csv - Contains GPT-4o judgments and scores for evaluated models

Usage:
    python generate_csv_from_data.py --captions-dir reports/minseok_prompt_audio_x_caption_sorted --evaluations-dir reports/caption_full_comparison_minseok_audio_x_caption_sorted_completeness --output-dir csv_output
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VLMDataCSVGenerator:
    """Generate CSV files from VLM caption data and evaluations."""
    
    def __init__(self, captions_dir: str, evaluations_dir: str, output_dir: str):
        self.captions_dir = Path(captions_dir)
        self.evaluations_dir = Path(evaluations_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model order for columns
        self.model_names = ["vila-1.5", "nvila", "cosmos_reason1", "qwen3-vl-30b-a3b-instruct", "gemini-2.5-pro"]
        self.evaluated_models = ["vila-1.5", "nvila", "cosmos_reason1", "qwen3-vl-30b-a3b-instruct"]  # Exclude ground truth
        
    def discover_videos_and_chunks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover all videos and their chunks from the captions directory."""
        
        video_chunk_data = {}
        
        # Use gemini-2.5-pro as reference for video structure (ground truth model)
        reference_model_dir = self.captions_dir / "gemini-2.5-pro"
        
        if not reference_model_dir.exists():
            logger.error(f"❌ Reference model directory not found: {reference_model_dir}")
            return {}
        
        for video_dir in reference_model_dir.iterdir():
            if not video_dir.is_dir():
                continue
                
            video_id = video_dir.name
            caption_file = video_dir / f"vlm_captions_{video_id}.json"
            
            if not caption_file.exists():
                logger.warning(f"⚠️  Caption file not found: {caption_file}")
                continue
            
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                chunks = data.get("chunk_responses", [])
                video_chunk_data[video_id] = chunks
                logger.debug(f"📄 Found {len(chunks)} chunks for video {video_id}")
                
            except Exception as e:
                logger.error(f"❌ Error loading caption file {caption_file}: {str(e)}")
                continue
        
        logger.info(f"🎬 Discovered {len(video_chunk_data)} videos")
        return video_chunk_data
    
    def load_model_captions(self, video_id: str, model_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load caption chunks for a specific model and video."""
        
        model_dir = self.captions_dir / model_name / video_id
        caption_file = model_dir / f"vlm_captions_{video_id}.json"
        
        if not caption_file.exists():
            logger.warning(f"⚠️  Caption file not found: {caption_file}")
            return None
        
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get("chunk_responses", [])
            
        except Exception as e:
            logger.error(f"❌ Error loading caption file {caption_file}: {str(e)}")
            return None
    
    def load_evaluations(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Load GPT-4o evaluation results from the first run (for qualitative analysis)."""
        
        # Use multi-run file to get individual judgments instead of averaged ones
        evaluation_file = self.evaluations_dir / "multi_run_caption_evaluations.json"
        
        if not evaluation_file.exists():
            logger.error(f"❌ Multi-run evaluation file not found: {evaluation_file}")
            return {}
        
        try:
            with open(evaluation_file, 'r', encoding='utf-8') as f:
                multi_run_data = json.load(f)
            
            # Extract evaluations from the first run for qualitative analysis
            if "all_runs" not in multi_run_data or not multi_run_data["all_runs"]:
                logger.error("❌ No runs found in multi-run data")
                return {}
            
            first_run = multi_run_data["all_runs"][0]
            evaluations = first_run.get("evaluations", [])
            
            logger.info(f"📊 Using evaluations from run {first_run.get('run_number', 1)} for qualitative analysis")
            
            # Organize evaluations by model_name -> video_id -> chunk_index
            organized_evals = {}
            
            for eval_data in evaluations:
                if "error" in eval_data:
                    continue
                
                model_name = eval_data["model_name"]
                video_id = eval_data["video_id"]
                chunk_index = eval_data["chunk_index"]
                
                if model_name not in organized_evals:
                    organized_evals[model_name] = {}
                if video_id not in organized_evals[model_name]:
                    organized_evals[model_name][video_id] = {}
                
                organized_evals[model_name][video_id][chunk_index] = eval_data
            
            logger.info(f"📊 Loaded evaluations for {len(organized_evals)} models from first run")
            return organized_evals
            
        except Exception as e:
            logger.error(f"❌ Error loading evaluation file: {str(e)}")
            return {}
    
    def generate_captions_csv(self) -> str:
        """Generate CSV file with VLM captions."""
        
        logger.info("📝 Generating VLM captions CSV...")
        
        # Discover video structure
        video_chunk_data = self.discover_videos_and_chunks()
        
        if not video_chunk_data:
            logger.error("❌ No video data found")
            return ""
        
        # Prepare data rows
        rows = []
        
        for video_id, reference_chunks in video_chunk_data.items():
            logger.info(f"🎬 Processing video {video_id} ({len(reference_chunks)} chunks)")
            
            # Load captions for all models
            model_captions = {}
            for model_name in self.model_names:
                chunks = self.load_model_captions(video_id, model_name)
                if chunks:
                    model_captions[model_name] = {chunk["chunk_id"]: chunk["content"] for chunk in chunks}
                else:
                    model_captions[model_name] = {}
            
            # Create rows for each chunk
            for chunk in reference_chunks:
                chunk_id = chunk["chunk_id"]
                start_time = chunk["start_time"]
                end_time = chunk["end_time"]
                
                row = {
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "start_time": start_time,
                    "end_time": end_time
                }
                
                # Add captions for each model
                for model_name in self.model_names:
                    caption = model_captions[model_name].get(chunk_id, "")
                    row[model_name] = caption
                
                rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        
        # Sort by video_id and chunk_id for consistency
        df = df.sort_values(["video_id", "chunk_id"])
        
        output_file = self.output_dir / "vlm_captions.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"💾 Saved VLM captions CSV: {output_file} ({len(df)} rows)")
        return str(output_file)
    
    def generate_evaluations_csv(self) -> str:
        """Generate CSV file with GPT-4o evaluations."""
        
        logger.info("📊 Generating GPT-4o evaluations CSV...")
        
        # Discover video structure
        video_chunk_data = self.discover_videos_and_chunks()
        
        if not video_chunk_data:
            logger.error("❌ No video data found")
            return ""
        
        # Load evaluation data
        evaluations = self.load_evaluations()
        
        if not evaluations:
            logger.error("❌ No evaluation data found")
            return ""
        
        # Prepare data rows
        rows = []
        
        for video_id, reference_chunks in video_chunk_data.items():
            logger.info(f"🎬 Processing evaluations for video {video_id} ({len(reference_chunks)} chunks)")
            
            # Create rows for each chunk
            for chunk in reference_chunks:
                chunk_id = chunk["chunk_id"]
                start_time = chunk["start_time"]
                end_time = chunk["end_time"]
                
                row = {
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "start_time": start_time,
                    "end_time": end_time
                }
                
                # Add evaluations for each model (judgment and score)
                for model_name in self.evaluated_models:
                    judgment = ""
                    score = ""
                    
                    if (model_name in evaluations and 
                        video_id in evaluations[model_name] and 
                        chunk_id in evaluations[model_name][video_id]):
                        
                        eval_data = evaluations[model_name][video_id][chunk_id]
                        judgment = eval_data.get("judgment", "")
                        score = eval_data.get("score", eval_data.get("overall_score", ""))
                    
                    row[model_name] = judgment
                    row[f"{model_name}_score"] = score
                
                rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        
        # Sort by video_id and chunk_id for consistency
        df = df.sort_values(["video_id", "chunk_id"])
        
        output_file = self.output_dir / "gpt4o_evaluations.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"💾 Saved GPT-4o evaluations CSV: {output_file} ({len(df)} rows)")
        return str(output_file)
    
    def generate_both_csvs(self) -> Dict[str, str]:
        """Generate both CSV files."""
        
        logger.info("🚀 Starting CSV generation...")
        
        results = {}
        
        # Generate captions CSV
        captions_file = self.generate_captions_csv()
        if captions_file:
            results["captions"] = captions_file
        
        # Generate evaluations CSV  
        evaluations_file = self.generate_evaluations_csv()
        if evaluations_file:
            results["evaluations"] = evaluations_file
        
        logger.info(f"✅ CSV generation complete! Generated {len(results)} files")
        
        return results
    
    def print_summary(self):
        """Print summary of available data."""
        
        logger.info("\n" + "="*60)
        logger.info("📊 VLM DATA SUMMARY")
        logger.info("="*60)
        
        logger.info(f"📂 Captions directory: {self.captions_dir}")
        logger.info(f"📂 Evaluations directory: {self.evaluations_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        # Check available models
        available_models = []
        for model_name in self.model_names:
            model_dir = self.captions_dir / model_name
            if model_dir.exists():
                available_models.append(model_name)
        
        logger.info(f"🤖 Available models: {available_models}")
        
        # Check videos
        video_chunk_data = self.discover_videos_and_chunks()
        if video_chunk_data:
            total_chunks = sum(len(chunks) for chunks in video_chunk_data.values())
            logger.info(f"🎬 Videos found: {len(video_chunk_data)}")
            logger.info(f"📊 Total chunks: {total_chunks}")
            
            for video_id, chunks in video_chunk_data.items():
                logger.info(f"  📹 {video_id}: {len(chunks)} chunks")
        
        # Check evaluations
        evaluations = self.load_evaluations()
        if evaluations:
            logger.info(f"📊 Models with evaluations (from first run): {list(evaluations.keys())}")
            for model_name, model_evals in evaluations.items():
                total_evals = sum(len(video_evals) for video_evals in model_evals.values())
                logger.info(f"  🤖 {model_name}: {total_evals} evaluations")
        
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV files from VLM caption data and evaluations"
    )
    parser.add_argument(
        "--captions-dir",
        required=True,
        help="Directory containing VLM caption results"
    )
    parser.add_argument(
        "--evaluations-dir",
        required=True,
        help="Directory containing GPT-4o evaluation results"
    )
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--csv-type",
        choices=["both", "captions", "evaluations"],
        default="both",
        help="Type of CSV to generate (default: both)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print data summary, don't generate CSV files"
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not Path(args.captions_dir).exists():
        logger.error(f"❌ Captions directory not found: {args.captions_dir}")
        return 1
    
    if not Path(args.evaluations_dir).exists():
        logger.error(f"❌ Evaluations directory not found: {args.evaluations_dir}")
        return 1
    
    try:
        # Initialize CSV generator
        csv_gen = VLMDataCSVGenerator(args.captions_dir, args.evaluations_dir, args.output_dir)
        
        # Print data summary
        csv_gen.print_summary()
        
        if args.summary_only:
            logger.info("📊 Summary complete (no CSV files generated)")
            return 0
        
        # Generate CSV files
        if args.csv_type == "both":
            results = csv_gen.generate_both_csvs()
            logger.info(f"\n📄 Generated CSV files:")
            for csv_type, file_path in results.items():
                logger.info(f"  📊 {csv_type}: {Path(file_path).name}")
        elif args.csv_type == "captions":
            captions_file = csv_gen.generate_captions_csv()
            if captions_file:
                logger.info(f"\n📄 Generated captions CSV: {Path(captions_file).name}")
        elif args.csv_type == "evaluations":
            evaluations_file = csv_gen.generate_evaluations_csv()
            if evaluations_file:
                logger.info(f"\n📄 Generated evaluations CSV: {Path(evaluations_file).name}")
        
        logger.info(f"\n📁 All CSV files saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ CSV generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    sys.exit(main())