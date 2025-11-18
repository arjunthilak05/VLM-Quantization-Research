"""
Analysis and Visualization Script
Generate publication-quality figures and tables from experimental results
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils import load_results, print_header

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultsAnalyzer:
    """Analyzer for experimental results"""

    def __init__(self, results_file: str, baseline_file: str, output_dir: str = "results"):
        self.results_file = results_file
        self.baseline_file = baseline_file
        self.output_dir = output_dir

        # Create output directories
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/tables", exist_ok=True)

        # Load results
        print_header("Loading Results")
        self.results = load_results(results_file)
        print(f"Loaded {len(self.results)} experimental results")

        if os.path.exists(baseline_file):
            self.baseline = load_results(baseline_file)
            print(f"Loaded {len(self.baseline)} baseline results")
        else:
            print(f"Warning: Baseline file not found: {baseline_file}")
            self.baseline = {}

        # Convert to DataFrame
        self.df = self.create_dataframe()

    def create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        print_header("Creating DataFrame")

        rows = []

        # Add baseline results
        for model_name, baseline_data in self.baseline.items():
            if isinstance(baseline_data, dict):
                row = {
                    'model': model_name,
                    'method': 'baseline',
                    'bits': 16,
                    'avg_accuracy': baseline_data.get('avg_accuracy', 0),
                }

                # Add efficiency metrics
                if 'efficiency' in baseline_data:
                    eff = baseline_data['efficiency']
                    row.update({
                        'latency_ms': eff.get('latency_ms', 0),
                        'model_size_mb': eff.get('model_size_mb', 0),
                        'peak_memory_mb': eff.get('peak_memory_mb', 0),
                        'throughput': eff.get('throughput_imgs_per_sec', 0)
                    })

                rows.append(row)

        # Add quantized results
        for result in self.results:
            if isinstance(result, dict) and result.get('status') == 'success':
                row = {
                    'model': result.get('model_name', 'unknown'),
                    'method': result.get('method', 'unknown'),
                    'bits': result.get('bits', 4),
                    'avg_accuracy': result.get('avg_accuracy', 0),
                    'latency_ms': result.get('latency_ms', 0),
                    'model_size_mb': result.get('model_size_mb', 0),
                    'peak_memory_mb': result.get('peak_memory_mb', 0),
                    'throughput': result.get('throughput_imgs_per_sec', 0),
                    'accuracy_recovery': result.get('accuracy_recovery_%', 0),
                    'speedup': result.get('speedup', 0),
                    'compression_ratio': result.get('compression_ratio', 0)
                }

                rows.append(row)

        df = pd.DataFrame(rows)

        print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        print(f"Methods: {df['method'].unique()}")
        print(f"Models: {df['model'].unique()}")

        return df

    def plot_accuracy_vs_model_size(self):
        """Plot accuracy vs model size scatter plot"""
        print("Creating accuracy vs model size plot...")

        fig, ax = plt.subplots(figsize=(10, 7))

        methods = self.df['method'].unique()
        colors = sns.color_palette("husl", len(methods))

        for method, color in zip(methods, colors):
            subset = self.df[self.df['method'] == method]

            ax.scatter(
                subset['model_size_mb'],
                subset['avg_accuracy'],
                label=method.upper(),
                s=150,
                alpha=0.7,
                color=color,
                edgecolors='black',
                linewidths=1.5
            )

            # Add labels for each point
            for _, row in subset.iterrows():
                ax.annotate(
                    f"{row['bits']}-bit",
                    (row['model_size_mb'], row['avg_accuracy']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )

        ax.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy vs Model Size Tradeoff', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = f"{self.output_dir}/plots/accuracy_vs_size.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved to {output_path}")

    def plot_accuracy_vs_latency(self):
        """Plot accuracy vs latency scatter plot"""
        print("Creating accuracy vs latency plot...")

        fig, ax = plt.subplots(figsize=(10, 7))

        methods = self.df['method'].unique()
        colors = sns.color_palette("husl", len(methods))

        for method, color in zip(methods, colors):
            subset = self.df[self.df['method'] == method]

            ax.scatter(
                subset['latency_ms'],
                subset['avg_accuracy'],
                label=method.upper(),
                s=150,
                alpha=0.7,
                color=color,
                edgecolors='black',
                linewidths=1.5
            )

        ax.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy vs Inference Latency', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = f"{self.output_dir}/plots/accuracy_vs_latency.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved to {output_path}")

    def plot_accuracy_recovery(self):
        """Plot accuracy recovery by method"""
        print("Creating accuracy recovery plot...")

        # Filter out baseline
        df_quant = self.df[self.df['method'] != 'baseline']

        if len(df_quant) == 0:
            print("  No quantized results to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by method
        df_grouped = df_quant.groupby('method')['accuracy_recovery'].mean().sort_values(ascending=False)

        colors = sns.color_palette("viridis", len(df_grouped))
        bars = ax.bar(range(len(df_grouped)), df_grouped.values, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Accuracy Recovery (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Quantization Method', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Recovery vs Baseline (Higher is Better)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(df_grouped)))
        ax.set_xticklabels([m.upper() for m in df_grouped.index], fontsize=10)
        ax.axhline(y=100, color='r', linestyle='--', label='Baseline (100%)', linewidth=2)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = f"{self.output_dir}/plots/accuracy_recovery.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved to {output_path}")

    def plot_multi_metric_comparison(self):
        """Plot multi-metric radar chart"""
        print("Creating multi-metric comparison plot...")

        # Filter out baseline and get quantized results
        df_quant = self.df[self.df['method'] != 'baseline']

        if len(df_quant) == 0:
            print("  No quantized results to plot")
            return

        # Calculate normalized metrics
        df_norm = df_quant.copy()

        # Normalize to 0-100 scale
        metrics = ['accuracy_recovery', 'speedup', 'compression_ratio']
        for metric in metrics:
            if metric in df_norm.columns and df_norm[metric].max() > 0:
                df_norm[f'{metric}_norm'] = (df_norm[metric] / df_norm[metric].max()) * 100

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 7))

        methods = df_quant['method'].unique()
        x = np.arange(len(methods))
        width = 0.25

        for i, metric in enumerate(metrics):
            norm_metric = f'{metric}_norm'
            if norm_metric in df_norm.columns:
                values = [df_norm[df_norm['method'] == m][norm_metric].mean() for m in methods]
                ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())

        ax.set_ylabel('Normalized Score (0-100)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Quantization Method', fontsize=12, fontweight='bold')
        ax.set_title('Multi-Metric Comparison (Normalized)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = f"{self.output_dir}/plots/multi_metric_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved to {output_path}")

    def plot_per_benchmark_comparison(self):
        """Plot per-benchmark accuracy comparison"""
        print("Creating per-benchmark comparison plot...")

        # This would require benchmark-level data from results
        # Placeholder for future implementation
        print("  Skipping (requires per-benchmark data)")

    def create_summary_table(self):
        """Create summary table of all results"""
        print("Creating summary table...")

        # Select key columns
        columns = ['model', 'method', 'bits', 'avg_accuracy', 'latency_ms',
                  'model_size_mb', 'accuracy_recovery', 'speedup', 'compression_ratio']

        available_cols = [c for c in columns if c in self.df.columns]
        summary_df = self.df[available_cols].copy()

        # Round numerical columns
        for col in summary_df.columns:
            if summary_df[col].dtype in ['float64', 'float32']:
                summary_df[col] = summary_df[col].round(2)

        # Save as CSV
        csv_path = f"{self.output_dir}/tables/summary_results.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"  Saved CSV to {csv_path}")

        # Save as LaTeX
        latex_table = summary_df.to_latex(
            index=False,
            caption="Summary of Quantization Results",
            label="tab:quantization_results",
            float_format="%.2f"
        )

        latex_path = f"{self.output_dir}/tables/summary_results.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"  Saved LaTeX to {latex_path}")

        return summary_df

    def generate_markdown_report(self):
        """Generate markdown report with key findings"""
        print("Generating markdown report...")

        report = []
        report.append("# VLM Quantization Research Results\n")
        report.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("\n## Summary Statistics\n")
        report.append(f"- Total models evaluated: {len(self.df)}")
        report.append(f"- Quantization methods: {', '.join(self.df['method'].unique())}")
        report.append(f"- Base models: {', '.join(self.df['model'].unique())}\n")

        report.append("\n## Key Findings\n")

        # Best accuracy recovery
        df_quant = self.df[self.df['method'] != 'baseline']
        if 'accuracy_recovery' in df_quant.columns and len(df_quant) > 0:
            best_acc = df_quant.loc[df_quant['accuracy_recovery'].idxmax()]
            report.append(f"### Best Accuracy Recovery")
            report.append(f"- Method: **{best_acc['method'].upper()}**")
            report.append(f"- Model: {best_acc['model']}")
            report.append(f"- Accuracy Recovery: **{best_acc['accuracy_recovery']:.2f}%**\n")

        # Best compression
        if 'compression_ratio' in df_quant.columns and len(df_quant) > 0:
            best_comp = df_quant.loc[df_quant['compression_ratio'].idxmax()]
            report.append(f"### Best Compression")
            report.append(f"- Method: **{best_comp['method'].upper()}**")
            report.append(f"- Model: {best_comp['model']}")
            report.append(f"- Compression Ratio: **{best_comp['compression_ratio']:.2f}x**\n")

        # Best speedup
        if 'speedup' in df_quant.columns and len(df_quant) > 0:
            best_speed = df_quant.loc[df_quant['speedup'].idxmax()]
            report.append(f"### Best Speedup")
            report.append(f"- Method: **{best_speed['method'].upper()}**")
            report.append(f"- Model: {best_speed['model']}")
            report.append(f"- Speedup: **{best_speed['speedup']:.2f}x**\n")

        # Method comparison
        report.append("\n## Method Comparison\n")
        report.append("| Method | Avg Accuracy Recovery | Avg Speedup | Avg Compression |\n")
        report.append("|--------|----------------------|-------------|----------------|\n")

        for method in df_quant['method'].unique():
            subset = df_quant[df_quant['method'] == method]
            acc_rec = subset['accuracy_recovery'].mean() if 'accuracy_recovery' in subset.columns else 0
            speedup = subset['speedup'].mean() if 'speedup' in subset.columns else 0
            comp = subset['compression_ratio'].mean() if 'compression_ratio' in subset.columns else 0

            report.append(f"| {method.upper()} | {acc_rec:.2f}% | {speedup:.2f}x | {comp:.2f}x |\n")

        report.append("\n## Visualizations\n")
        report.append("- [Accuracy vs Model Size](plots/accuracy_vs_size.png)\n")
        report.append("- [Accuracy vs Latency](plots/accuracy_vs_latency.png)\n")
        report.append("- [Accuracy Recovery](plots/accuracy_recovery.png)\n")
        report.append("- [Multi-Metric Comparison](plots/multi_metric_comparison.png)\n")

        # Save report
        report_path = f"{self.output_dir}/analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(''.join(report))

        print(f"  Saved report to {report_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print_header("Running Full Analysis", char="#")

        # Create all plots
        self.plot_accuracy_vs_model_size()
        self.plot_accuracy_vs_latency()
        self.plot_accuracy_recovery()
        self.plot_multi_metric_comparison()

        # Create tables
        self.create_summary_table()

        # Generate report
        self.generate_markdown_report()

        print_header("ANALYSIS COMPLETE", char="#")
        print(f"All outputs saved to: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze quantization results and generate visualizations")
    parser.add_argument("--results", type=str, default="results/comprehensive_results.json",
                        help="Path to comprehensive results file")
    parser.add_argument("--baseline", type=str, default="results/baseline_results.json",
                        help="Path to baseline results file")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for plots and tables")

    args = parser.parse_args()

    # Check if results exist
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        print("Please run comprehensive evaluation first (06_comprehensive_eval.py)")
        return

    # Run analysis
    analyzer = ResultsAnalyzer(args.results, args.baseline, args.output)
    analyzer.run_full_analysis()

    # Print summary
    print("\nGenerated files:")
    print(f"  - Plots: {args.output}/plots/")
    print(f"  - Tables: {args.output}/tables/")
    print(f"  - Report: {args.output}/analysis_report.md")


if __name__ == "__main__":
    main()
