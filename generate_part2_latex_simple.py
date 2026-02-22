#!/usr/bin/env python3
"""
Generate simpler LaTeX code for Part 2 accuracy comparison graph.
Creates a version using basic TikZ (no pgfplots dependency) or a table version.
"""

import csv
import os
from collections import defaultdict

def load_part2_data(csv_file):
    """Load Part 2 experiment data from CSV."""
    experiments = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Part 2' in row['Part']:
                exp_name = row['Experiment']
                experiments[exp_name].append({
                    'iteration': int(row['Iteration']),
                    'accuracy': float(row['Accuracy (%)']),
                    'f1': float(row['F1']) if row['F1'] else None,
                })
    
    # Sort by iteration for each experiment
    for exp_name in experiments:
        experiments[exp_name].sort(key=lambda x: x['iteration'])
        # Remove duplicates
        seen = set()
        unique_data = []
        for d in experiments[exp_name]:
            if d['iteration'] not in seen:
                unique_data.append(d)
                seen.add(d['iteration'])
        experiments[exp_name] = unique_data
    
    return experiments

def get_clean_experiment_name(exp_name):
    """Get a cleaner name for display."""
    name_map = {
        'REDO_insqrt': 'Inverse Sqrt',
        'REDO_balnormal': 'Balanced Normal',
        'REDO_log': 'Log',
        'REDO_smooth': 'Smooth',
        'REDO_uniform': 'Uniform'
    }
    return name_map.get(exp_name, exp_name.replace('REDO_', ''))

def get_class_weights(exp_name):
    """Get class weights for an experiment."""
    weights_map = {
        'REDO_insqrt': [0.928, 0.947, 1.125],
        'REDO_balnormal': [0.854, 0.890, 1.256],
        'REDO_log': [0.923, 0.944, 1.133],
        'REDO_smooth': [1.130, 1.087, 0.783],
        'REDO_uniform': [1.000, 1.000, 1.000]
    }
    return weights_map.get(exp_name, [1.0, 1.0, 1.0])

def get_legend_label(exp_name):
    """Get legend label with class weights."""
    clean_name = get_clean_experiment_name(exp_name)
    weights = get_class_weights(exp_name)
    weights_str = f"[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
    return f"{clean_name} {weights_str}"

def generate_table_version(experiments, output_file='results_visualization/part2_accuracy_table.tex'):
    """Generate a LaTeX table version showing key metrics."""
    
    latex_code = """% Part 2: Class Weights - Summary Table
% Simple LaTeX table version (no special packages needed)

\\begin{table}[htbp]
\\centering
\\caption{Part 2: Class Weights - Performance Summary}
\\label{tab:part2_class_weights}
\\begin{tabular}{lcccccc}
\\toprule
Strategy & Class Weights & Best Acc & Best Acc & Mean Acc & Best F1 & Mean F1 \\\\
 & [Frail, Prefrail, Nonfrail] & (\\%) & Iter & (\\%) & & \\\\
\\midrule
"""
    
    for exp_name, data in sorted(experiments.items()):
        if not data:
            continue
        
        clean_name = get_clean_experiment_name(exp_name)
        weights = get_class_weights(exp_name)
        weights_str = f"[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
        
        best_acc = max(d['accuracy'] for d in data)
        best_acc_iter = next(d['iteration'] for d in data if d['accuracy'] == best_acc)
        mean_acc = sum(d['accuracy'] for d in data) / len(data)
        
        best_f1 = max(d['f1'] for d in data if d['f1'] is not None) if any(d['f1'] for d in data) else None
        mean_f1 = sum(d['f1'] for d in data if d['f1'] is not None) / len([d for d in data if d['f1'] is not None]) if any(d['f1'] for d in data) else None
        
        f1_str = f"{best_f1:.3f}" if best_f1 else "N/A"
        mean_f1_str = f"{mean_f1:.3f}" if mean_f1 else "N/A"
        
        latex_code += f"{clean_name} & {weights_str} & {best_acc:.2f} & {best_acc_iter} & {mean_acc:.2f} & {f1_str} & {mean_f1_str} \\\\\n"
    
    latex_code += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_code)
    
    print(f"✓ Saved table version: {output_file}")
    return output_file

def generate_simple_tikz_version(experiments, output_file='results_visualization/part2_accuracy_simple.tex'):
    """Generate a simpler TikZ version (minimal dependencies)."""
    
    # Find max values for scaling
    max_iter = 0
    max_acc = 0
    for data in experiments.values():
        if data:
            max_iter = max(max_iter, max(d['iteration'] for d in data))
            max_acc = max(max_acc, max(d['accuracy'] for d in data))
    
    # Scale factors
    x_scale = 12.0 / max_iter  # 12cm width
    y_scale = 7.0 / 100.0      # 7cm height for 0-100%
    
    colors = {
        'REDO_insqrt': 'blue',
        'REDO_balnormal': 'red',
        'REDO_log': 'green!60!black',
        'REDO_smooth': 'orange',
        'REDO_uniform': 'purple'
    }
    
    markers = {
        'REDO_insqrt': 'circle',
        'REDO_balnormal': 'square',
        'REDO_log': 'triangle',
        'REDO_smooth': 'diamond',
        'REDO_uniform': 'star'
    }
    
    latex_code = f"""% Part 2: Class Weights - Accuracy Comparison (Simple TikZ)
% Minimal dependencies: just tikz package

\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}[scale=1]
    % Draw axes
    \\draw[->, thick] (0,0) -- ({max_iter * x_scale + 1}, 0) node[right] {{Iteration}};
    \\draw[->, thick] (0,0) -- (0, 8) node[above] {{Accuracy (\\%)}};
    
    % Draw grid
    \\draw[gray!30, dashed] (0,0) grid ({max_iter * x_scale}, 7);
    
    % Draw y-axis labels
    \\foreach \\y in {{0, 20, 40, 60, 80, 100}} {{
        \\draw (0, \\y/100*7) -- (-0.1, \\y/100*7) node[left] {{\\y}};
    }}
    
    % Draw x-axis labels (every 2000 iterations)
    \\foreach \\x in {{0, 2000, 4000, 6000, 8000, 10000}} {{
        \\ifnum\\x<{max_iter}
            \\draw (\\x*{x_scale}, 0) -- (\\x*{x_scale}, -0.1) node[below] {{\\x}};
        \\fi
    }}
    
    % Title
    \\node[above] at ({max_iter * x_scale / 2}, 7.5) {{\\textbf{{Part 2: Class Weights - Accuracy Comparison}}}};
    
"""
    
    # Draw each experiment
    for i, (exp_name, data) in enumerate(sorted(experiments.items())):
        if not data:
            continue
        
        color = colors.get(exp_name, 'black')
        marker = markers.get(exp_name, 'circle')
        legend_label = get_legend_label(exp_name).replace('_', '\\_')
        
        # Draw line
        coords = ' '.join([f"({d['iteration']*x_scale},{d['accuracy']*y_scale})" for d in data])
        latex_code += f"    % {get_clean_experiment_name(exp_name)}\n"
        latex_code += f"    \\draw[{color}, thick] plot[smooth] coordinates {{{coords}}};\n"
        
        # Draw markers
        for d in data[::2]:  # Every other point to avoid clutter
            latex_code += f"    \\node[{color}, fill=white, {marker}, draw={color}, inner sep=1.5pt] at ({d['iteration']*x_scale},{d['accuracy']*y_scale}) {{}};\n"
        
        # Legend entry
        y_pos = 6.5 - i * 0.4
        latex_code += f"    \\node[{color}, {marker}, draw={color}, fill=white, inner sep=2pt] at (0.5, {y_pos}) {{}};\n"
        latex_code += f"    \\node[right] at (0.7, {y_pos}) {{\\small {legend_label}}};\n\n"
    
    latex_code += """\\end{tikzpicture}
\\caption{Accuracy comparison across different class weighting strategies during training.}
\\label{fig:part2_accuracy_comparison}
\\end{figure}
"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_code)
    
    print(f"✓ Saved simple TikZ version: {output_file}")
    return output_file

def generate_minimal_pgfplots(experiments, output_file='results_visualization/part2_accuracy_minimal.tex'):
    """Generate minimal pgfplots version with embedded data."""
    
    # Find max iteration
    max_iter = 0
    for data in experiments.values():
        if data:
            max_iter = max(max_iter, max(d['iteration'] for d in data))
    
    latex_code = """% Part 2: Class Weights - Accuracy Comparison (Minimal PGFPlots)
% Requires: \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}

\\begin{figure}[htbp]
\\centering
\\begin{tikzpicture}
\\begin{axis}[
    width=0.9\\textwidth,
    height=6cm,
    xlabel=Iteration,
    ylabel=Accuracy (\\%),
    title={Part 2: Class Weights - Accuracy Comparison Over Training},
    xmin=0,
    xmax=""" + str(int(max_iter * 1.02)) + """,
    ymin=0,
    ymax=100,
    grid=major,
    legend pos=south east,
    legend style={font=\\small}
]

"""
    
    colors = ['blue', 'red', 'green!60!black', 'orange', 'purple']
    markers = ['*', 'square*', 'triangle*', 'diamond*', 'pentagon*']
    
    for i, (exp_name, data) in enumerate(sorted(experiments.items())):
        if not data:
            continue
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        legend_label = get_legend_label(exp_name).replace('_', '\\_')
        
        # Create coordinate string
        coords = ' '.join([f"({d['iteration']},{d['accuracy']})" for d in data])
        
        latex_code += f"\\addplot[color={color}, mark={marker}, mark size=1.5pt, thick] coordinates {{\n"
        latex_code += f"    {coords}\n"
        latex_code += f"}};\n"
        latex_code += f"\\addlegendentry{{{legend_label}}}\n\n"
    
    latex_code += """\\end{axis}
\\end{tikzpicture}
\\caption{Accuracy comparison across different class weighting strategies during training. 
Class weights are shown in the legend as [Frail, Prefrail, Nonfrail].}
\\label{fig:part2_accuracy_comparison}
\\end{figure}
"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_code)
    
    print(f"✓ Saved minimal pgfplots version: {output_file}")
    return output_file

def main():
    csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    print("=" * 80)
    print("GENERATING ALTERNATIVE LATEX VERSIONS FOR PART 2")
    print("=" * 80)
    
    experiments = load_part2_data(csv_file)
    
    if not experiments:
        print("\n❌ No Part 2 data found in CSV!")
        return
    
    print(f"\nFound {len(experiments)} Part 2 experiment(s)\n")
    
    # Generate all versions
    table_file = generate_table_version(experiments)
    simple_file = generate_simple_tikz_version(experiments)
    minimal_file = generate_minimal_pgfplots(experiments)
    
    print("\n" + "=" * 80)
    print("✓ All LaTeX versions generated!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  1. {table_file} - Simple table (no special packages)")
    print(f"  2. {simple_file} - Basic TikZ (just \\usepackage{{tikz}})")
    print(f"  3. {minimal_file} - Minimal pgfplots (simplified)")
    print("\nChoose based on your needs:")
    print("  - Table: Simplest, no graphics packages needed")
    print("  - Simple TikZ: Basic graphics, minimal dependencies")
    print("  - Minimal PGFPlots: Clean pgfplots version, easier than full version")
    print("=" * 80)

if __name__ == '__main__':
    main()

