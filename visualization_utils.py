#!/usr/bin/env python3
"""
Visualization Utils

Create comprehensive visualizations for VLM caption evaluation results including
radar plots, bar charts, heatmaps, and statistical distributions.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure matplotlib and seaborn
plt.style.use('default')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptionVisualizationUtils:
    """Comprehensive visualization utilities for caption evaluation results."""
    
    def __init__(self, output_dir: Path, prompt_type: str = "detailed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_type = prompt_type
        
        # Set up matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def create_radar_plot(self, summary_stats: pd.DataFrame) -> str:
        """Create radar plot comparing models across all criteria."""
        
        if summary_stats.empty:
            logger.warning("⚠️  No data for radar plot")
            return ""
        
        # Prepare data for radar plot
        criteria = ['factual_accuracy_mean', 'completeness_mean', 'clarity_mean', 
                   'conciseness_mean', 'temporal_accuracy_mean']
        criteria_labels = ['Factual\nAccuracy', 'Completeness', 'Clarity', 
                          'Conciseness', 'Temporal\nAccuracy']
        
        # Set up the radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each criterion
        angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(summary_stats)))
        
        for idx, (_, row) in enumerate(summary_stats.iterrows()):
            values = [row[criterion] for criterion in criteria]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=f"{row['model']} ({row['overall_mean']:.1f})", color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria_labels)
        ax.set_ylim(0, 10)
        ax.set_yticks(np.arange(0, 11, 2))
        ax.set_yticklabels([str(i) for i in range(0, 11, 2)])
        ax.grid(True)
        
        # Add title and legend
        plt.title('VLM Caption Evaluation - Multi-Criteria Comparison\n(Scores averaged across multiple runs)', 
                 size=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Save plot
        plot_file = self.output_dir / "radar_plot_comparison.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved radar plot: {plot_file}")
        return str(plot_file)
    
    def create_bar_chart_comparison(self, summary_stats: pd.DataFrame) -> str:
        """Create bar chart comparing overall scores with error bars."""
        
        if summary_stats.empty:
            logger.warning("⚠️  No data for bar chart")
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        models = summary_stats['model'].tolist()
        means = summary_stats['overall_mean'].tolist()
        stds = summary_stats['overall_std'].tolist()
        
        # Create bar chart
        bars = ax.bar(models, means, yerr=stds, capsize=5, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(models))),
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize plot
        ax.set_ylabel('Overall Score (1-10 scale)')
        ax.set_xlabel('Models')
        ax.set_title('VLM Caption Evaluation - Overall Score Comparison\n(Error bars show standard deviation across runs)', 
                    fontweight='bold', pad=20)
        ax.set_ylim(0, 10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                   f'{mean:.1f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "bar_chart_overall_scores.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved bar chart: {plot_file}")
        return str(plot_file)
    
    def create_criteria_heatmap(self, summary_stats: pd.DataFrame) -> str:
        """Create heatmap showing all criteria scores for all models."""
        
        if summary_stats.empty:
            logger.warning("⚠️  No data for heatmap")
            return ""
        
        # Prepare data for heatmap
        criteria_cols = ['factual_accuracy_mean', 'completeness_mean', 'clarity_mean',
                        'conciseness_mean', 'temporal_accuracy_mean', 'overall_mean']
        criteria_labels = ['Factual\nAccuracy', 'Completeness', 'Clarity', 
                          'Conciseness', 'Temporal\nAccuracy', 'Overall']
        
        # Create matrix for heatmap
        heatmap_data = summary_stats[['model'] + criteria_cols].copy()
        heatmap_data = heatmap_data.set_index('model')
        heatmap_data.columns = criteria_labels
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   vmin=0, vmax=10, center=5, cbar_kws={'label': 'Score (1-10)'})
        
        # Customize plot
        plt.title('VLM Caption Evaluation - Criteria Heatmap\n(Higher scores = better performance)', 
                 fontweight='bold', pad=20)
        plt.xlabel('Evaluation Criteria')
        plt.ylabel('Models')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Save plot
        plot_file = self.output_dir / "heatmap_criteria_scores.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved heatmap: {plot_file}")
        return str(plot_file)
    
    def create_std_deviation_plot(self, summary_stats: pd.DataFrame) -> str:
        """Create plot showing standard deviations to assess consistency."""
        
        if summary_stats.empty:
            logger.warning("⚠️  No data for std deviation plot")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Overall score std deviation
        models = summary_stats['model'].tolist()
        overall_stds = summary_stats['overall_std'].tolist()
        
        bars1 = ax1.bar(models, overall_stds, 
                       color=plt.cm.plasma(np.linspace(0, 1, len(models))),
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        ax1.set_ylabel('Standard Deviation')
        ax1.set_xlabel('Models')
        ax1.set_title('Model Consistency - Overall Score\n(Lower = more consistent)', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, std in zip(bars1, overall_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Criteria-wise std deviations
        criteria_std_cols = ['factual_accuracy_std', 'completeness_std', 'clarity_std',
                           'conciseness_std', 'temporal_accuracy_std']
        criteria_labels = ['Factual', 'Complete', 'Clarity', 'Concise', 'Temporal']
        
        # Prepare data for stacked bar chart
        std_data = summary_stats[criteria_std_cols].values
        
        # Create stacked bar chart
        bottom = np.zeros(len(models))
        colors = plt.cm.Set2(np.linspace(0, 1, len(criteria_std_cols)))
        
        for i, (criterion, color) in enumerate(zip(criteria_labels, colors)):
            ax2.bar(models, std_data[:, i], bottom=bottom, label=criterion, 
                   color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += std_data[:, i]
        
        ax2.set_ylabel('Cumulative Standard Deviation')
        ax2.set_xlabel('Models')
        ax2.set_title('Model Consistency - By Criteria\n(Lower = more consistent)', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels
        for ax in [ax1, ax2]:
            ax.set_xticklabels(models, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "consistency_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved consistency analysis: {plot_file}")
        return str(plot_file)
    
    def create_evaluation_distribution_plot(self, evaluations: List[Dict[str, Any]]) -> str:
        """Create distribution plots showing score distributions across models."""
        
        if not evaluations:
            logger.warning("⚠️  No evaluation data for distribution plot")
            return ""
        
        # Prepare data
        model_scores = {}
        for eval_result in evaluations:
            if "error" in eval_result:
                continue
                
            model = eval_result["model_name"]
            if model not in model_scores:
                model_scores[model] = {"overall": [], "factual": [], "completeness": [], 
                                     "clarity": [], "conciseness": [], "temporal": []}
            
            # Handle different prompt types flexibly
            prompt_type = self.prompt_type
            
            # Always append overall score
            model_scores[model]["overall"].append(eval_result["overall_score"])
            
            if prompt_type == "detailed":
                # Detailed prompt has all criteria
                model_scores[model]["factual"].append(eval_result["factual_accuracy"]["score"])
                model_scores[model]["completeness"].append(eval_result["completeness"]["score"])
                model_scores[model]["clarity"].append(eval_result["clarity"]["score"])
                model_scores[model]["conciseness"].append(eval_result["conciseness"]["score"])
                model_scores[model]["temporal"].append(eval_result["temporal_accuracy"]["score"])
            else:
                # For coverage_hallucination and completeness prompts, use overall score for all criteria
                score = eval_result.get("score", eval_result["overall_score"])
                model_scores[model]["factual"].append(score)
                model_scores[model]["completeness"].append(score)
                model_scores[model]["clarity"].append(score)
                model_scores[model]["conciseness"].append(score)
                model_scores[model]["temporal"].append(score)
        
        if not model_scores:
            logger.warning("⚠️  No valid scores for distribution plot")
            return ""
        
        # Create subplots for each criterion
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        criteria = ["overall", "factual", "completeness", "clarity", "conciseness", "temporal"]
        criterion_titles = ["Overall Score", "Factual Accuracy", "Completeness", 
                          "Clarity", "Conciseness", "Temporal Accuracy"]
        
        for idx, (criterion, title) in enumerate(zip(criteria, criterion_titles)):
            ax = axes[idx]
            
            # Prepare data for violin plot
            data_for_plot = []
            labels = []
            
            for model in model_scores.keys():
                scores = model_scores[model][criterion]
                if scores:
                    data_for_plot.append(scores)
                    labels.append(f"{model}\n(n={len(scores)})")
            
            if data_for_plot:
                # Create violin plot
                parts = ax.violinplot(data_for_plot, positions=range(len(labels)), showmeans=True)
                
                # Customize violin plot
                for pc in parts['bodies']:
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Score')
                ax.set_title(title)
                ax.set_ylim(0, 10)
                ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Score Distribution Analysis\n(Violin plots show score distribution across all evaluations)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "score_distributions.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved distribution analysis: {plot_file}")
        return str(plot_file)
    
    def create_temporal_accuracy_analysis(self, evaluations: List[Dict[str, Any]]) -> str:
        """Create specialized analysis for temporal accuracy performance."""
        
        if not evaluations:
            logger.warning("⚠️  No evaluation data for temporal analysis")
            return ""
        
        # Extract temporal accuracy data
        temporal_data = []
        for eval_result in evaluations:
            if "error" in eval_result or "temporal_accuracy" not in eval_result:
                continue
                
            temporal_data.append({
                "model": eval_result["model_name"],
                "video_id": eval_result["video_id"],
                "chunk_index": eval_result["chunk_index"],
                "temporal_score": eval_result["temporal_accuracy"]["score"],
                "start_time": float(eval_result["start_time"]),
                "end_time": float(eval_result["end_time"]),
                "duration": float(eval_result["end_time"]) - float(eval_result["start_time"])
            })
        
        if not temporal_data:
            logger.warning("⚠️  No valid temporal data for analysis")
            return ""
        
        df = pd.DataFrame(temporal_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Temporal accuracy by model (box plot)
        df.boxplot(column='temporal_score', by='model', ax=ax1)
        ax1.set_title('Temporal Accuracy by Model')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Temporal Accuracy Score')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 2: Temporal accuracy vs chunk duration
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax2.scatter(model_data['duration'], model_data['temporal_score'], 
                       label=model, alpha=0.6, s=30)
        
        ax2.set_xlabel('Chunk Duration (seconds)')
        ax2.set_ylabel('Temporal Accuracy Score')
        ax2.set_title('Temporal Accuracy vs Chunk Duration')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Average temporal accuracy by model
        model_means = df.groupby('model')['temporal_score'].agg(['mean', 'std']).reset_index()
        
        bars = ax3.bar(model_means['model'], model_means['mean'], 
                      yerr=model_means['std'], capsize=5,
                      color=plt.cm.viridis(np.linspace(0, 1, len(model_means))))
        
        ax3.set_ylabel('Average Temporal Accuracy')
        ax3.set_xlabel('Model')
        ax3.set_title('Average Temporal Accuracy by Model')
        ax3.set_xticklabels(model_means['model'], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars, model_means['mean'], model_means['std']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                    f'{mean:.1f}±{std:.2f}', ha='center', va='bottom')
        
        # Plot 4: Temporal accuracy distribution
        ax4.hist([df[df['model'] == model]['temporal_score'].values 
                 for model in df['model'].unique()], 
                label=df['model'].unique(), alpha=0.7, bins=20)
        ax4.set_xlabel('Temporal Accuracy Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Temporal Accuracy Score Distribution')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Temporal Accuracy Analysis\n(Specialized analysis for timestamp and temporal sequence evaluation)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "temporal_accuracy_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved temporal accuracy analysis: {plot_file}")
        return str(plot_file)
    
    def create_all_visualizations(self, summary_stats: pd.DataFrame, 
                                evaluations: List[Dict[str, Any]]) -> List[str]:
        """Create all visualization plots and return list of saved files."""
        
        logger.info("🎨 Creating comprehensive visualizations...")
        
        saved_files = []
        
        # Create all plots
        plots = [
            self.create_radar_plot(summary_stats),
            self.create_bar_chart_comparison(summary_stats),
            self.create_criteria_heatmap(summary_stats),
            self.create_std_deviation_plot(summary_stats),
            self.create_evaluation_distribution_plot(evaluations),
            self.create_temporal_accuracy_analysis(evaluations)
        ]
        
        # Filter out empty results
        saved_files = [plot for plot in plots if plot]
        
        logger.info(f"🎨 Created {len(saved_files)} visualization plots")
        
        return saved_files