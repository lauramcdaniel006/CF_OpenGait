#!/usr/bin/env python3
"""
Generate LaTeX code for Part 2 accuracy comparison graph using pgfplots.
This creates a .tex file that can be used directly in Overleaf.
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
                    'precision': float(row['Precision']) if row['Precision'] else None,
                    'recall': float(row['Recall']) if row['Recall'] else None
                })
    
    # Sort by iteration for each experiment
    for exp_name in experiments:
        experiments[exp_name].sort(key=lambda x: x['iteration'])
    
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
    # Format: "Name [Frail, Prefrail, Nonfrail]"
    weights_str = f"[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
    return f"{clean_name} {weights_str}"

def format_data_for_latex(experiments):
    """Format data for pgfplots table format."""
    data_blocks = {}
    
    for exp_name, data in sorted(experiments.items()):
        clean_name = get_clean_experiment_name(exp_name)
        # Create data table - remove duplicates by iteration
        seen_iterations = set()
        data_lines = []
        for d in data:
            if d['iteration'] not in seen_iterations:
                data_lines.append(f"            {d['iteration']} {d['accuracy']}")
                seen_iterations.add(d['iteration'])
        
        data_blocks[exp_name] = {
            'name': clean_name,
            'legend': get_legend_label(exp_name),
            'data': '\n'.join(data_lines)
        }
    
    return data_blocks

def generate_latex_code(experiments, output_file='results_visualization/part2_accuracy_comparison.tex'):
    """Generate LaTeX code for the graph."""
    
    data_blocks = format_data_for_latex(experiments)
    
    # Color definitions (pgfplots colors)
    colors = {
        'REDO_insqrt': 'blue',
        'REDO_balnormal': 'red',
        'REDO_log': 'green!60!black',
        'REDO_smooth': 'orange',
        'REDO_uniform': 'purple'
    }
    
    # Marker styles
    markers = {
        'REDO_insqrt': 'o',
        'REDO_balnormal': 'square',
        'REDO_log': 'triangle',
        'REDO_smooth': 'diamond',
        'REDO_uniform': 'pentagon'
    }
    
    # Line styles
    line_styles = {
        'REDO_insqrt': 'solid',
        'REDO_balnormal': 'dashed',
        'REDO_log': 'dashdotted',
        'REDO_smooth': 'dotted',
        'REDO_uniform': 'solid'
    }
    
    # Find max iteration for x-axis
    max_iter = 0
    for data in experiments.values():
        if data:
            max_iter = max(max_iter, max(d['iteration'] for d in data))
    
    # Generate LaTeX code
    latex_code = f"""\\documentclass{{article}}
\\usepackage{{pgfplots}}
\\usepackage{{times}}  % Times New Roman font
\\pgfplotsset{{compat=1.18}}
\\usepgfplotslibrary{{dateplot}}

\\begin{{document}}

\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=14cm,
    height=8cm,
    xlabel={{Iteration}},
    ylabel={{Accuracy (\\%)}},
    title={{Part 2: Class Weights - Accuracy Comparison Over Training}},
    xmin=0,
    xmax={max_iter * 1.02:.0f},
    ymin=0,
    ymax=100,
    grid=major,
    grid style={{dashed, gray!30}},
    legend style={{
        at={{(0.98,0.02)}},
        anchor=south east,
        cells={{anchor=west}},
        font=\\small,
        fill=white,
        draw=gray,
        rounded corners,
        shadow
    }},
    tick label style={{font=\\small}},
    label style={{font=\\bfseries}},
    title style={{font=\\bfseries}},
    axis background/.style={{fill=gray!5}},
    every axis plot/.append style={{line width=1.5pt, mark size=2pt}}
]

"""
    
    # Add each experiment
    for i, (exp_name, data) in enumerate(sorted(experiments.items())):
        exp_data = data_blocks[exp_name]
        color = colors.get(exp_name, 'black')
        marker = markers.get(exp_name, 'o')
        line_style = line_styles.get(exp_name, 'solid')
        legend_label = exp_data['legend']
        
        # Escape special characters in legend
        legend_label = legend_label.replace('_', '\\_')
        
        latex_code += f"""\\addplot[
    color={color},
    mark={marker},
    mark options={{fill=white, draw={color}, line width=0.8pt}},
    style={line_style},
    opacity=0.85
] table {{
{exp_data['data']}
}};
\\addlegendentry{{{legend_label}}}

"""
    
    # Find and annotate best points
    for exp_name, data in sorted(experiments.items()):
        if data:
            best_idx = max(range(len(data)), key=lambda i: data[i]['accuracy'])
            best_acc = data[best_idx]['accuracy']
            best_iter = data[best_idx]['iteration']
            color = colors.get(exp_name, 'black')
            
            latex_code += f"""\\node[
    pin={{[pin distance=5pt, pin edge={{->, {color}, line width=1pt}}]above right:{{\\textbf{{{best_acc:.1f}\\%}}}}}},
    fill={color},
    circle,
    inner sep=1.5pt
] at (axis cs:{best_iter},{best_acc}) {{}};

"""
    
    latex_code += """\\end{axis}
\\end{tikzpicture}
\\caption{Accuracy comparison across different class weighting strategies during training. 
Class weights are shown in the legend as [Frail, Prefrail, Nonfrail]. 
The best accuracy point for each strategy is annotated.}
\\label{fig:part2_accuracy_comparison}
\\end{figure}

\\end{document}
"""
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_code)
    
    print(f"✓ Saved LaTeX file: {output_file}")
    return output_file

def generate_standalone_latex(experiments, output_file='results_visualization/part2_accuracy_comparison_standalone.tex'):
    """Generate a standalone LaTeX file that can be included in Overleaf."""
    
    data_blocks = format_data_for_latex(experiments)
    
    # Color definitions (using RGB for better control)
    colors_rgb = {
        'REDO_insqrt': '0,0,255',      # Blue
        'REDO_balnormal': '255,0,0',   # Red
        'REDO_log': '0,128,0',         # Green
        'REDO_smooth': '255,165,0',    # Orange
        'REDO_uniform': '128,0,128'    # Purple
    }
    
    # Marker styles
    markers = {
        'REDO_insqrt': '*',
        'REDO_balnormal': 'square*',
        'REDO_log': 'triangle*',
        'REDO_smooth': 'diamond*',
        'REDO_uniform': 'pentagon*'
    }
    
    # Line styles
    line_styles = {
        'REDO_insqrt': '',
        'REDO_balnormal': 'dashed',
        'REDO_log': 'dashdotted',
        'REDO_smooth': 'dotted',
        'REDO_uniform': ''
    }
    
    # Find max iteration for x-axis
    max_iter = 0
    for data in experiments.values():
        if data:
            max_iter = max(max_iter, max(d['iteration'] for d in data))
    
    # Generate LaTeX code
    latex_code = f"""% Part 2: Class Weights - Accuracy Comparison Graph
% Generated for Overleaf
% Include this in your document with: \\input{{part2_accuracy_comparison_standalone.tex}}

\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=14cm,
    height=8cm,
    xlabel={{Iteration}},
    ylabel={{Accuracy (\\%)}},
    title={{Part 2: Class Weights - Accuracy Comparison Over Training}},
    xmin=0,
    xmax={max_iter * 1.02:.0f},
    ymin=0,
    ymax=100,
    grid=major,
    grid style={{dashed, gray!30}},
    legend style={{
        at={{(0.98,0.02)}},
        anchor=south east,
        cells={{anchor=west}},
        font=\\small,
        fill=white,
        draw=gray,
        rounded corners,
        shadow
    }},
    tick label style={{font=\\small}},
    label style={{font=\\bfseries}},
    title style={{font=\\bfseries}},
    axis background/.style={{fill=gray!5}},
    every axis plot/.append style={{line width=1.5pt, mark size=2pt}},
    scaled ticks=false,
    tick label style={{/pgf/number format/fixed}}
]

"""
    
    # Add each experiment
    for i, (exp_name, data) in enumerate(sorted(experiments.items())):
        exp_data = data_blocks[exp_name]
        color_rgb = colors_rgb.get(exp_name, '0,0,0')
        marker = markers.get(exp_name, '*')
        line_style = line_styles.get(exp_name, '')
        legend_label = exp_data['legend']
        
        # Escape special characters in legend
        legend_label = legend_label.replace('_', '\\_')
        
        # Build plot options
        plot_options = [
            f"color={{rgb,255:red,{color_rgb.split(',')[0]};green,{color_rgb.split(',')[1]};blue,{color_rgb.split(',')[2]}}}",
            f"mark={marker}",
            "mark options={fill=white, draw={rgb,255:red," + color_rgb.split(',')[0] + ";green," + color_rgb.split(',')[1] + ";blue," + color_rgb.split(',')[2] + "}, line width=0.8pt}",
            "opacity=0.85"
        ]
        
        if line_style:
            plot_options.append(f"style={line_style}")
        
        plot_opts_str = ',\n    '.join(plot_options)
        
        latex_code += f"""\\addplot[
    {plot_opts_str}
] table {{
{exp_data['data']}
}};
\\addlegendentry{{{legend_label}}}

"""
    
    # Find and annotate best points
    for exp_name, data in sorted(experiments.items()):
        if data:
            best_idx = max(range(len(data)), key=lambda i: data[i]['accuracy'])
            best_acc = data[best_idx]['accuracy']
            best_iter = data[best_idx]['iteration']
            color_rgb = colors_rgb.get(exp_name, '0,0,0')
            
            latex_code += f"""\\node[
    pin={{[pin distance=5pt]above right:{{\\textbf{{{best_acc:.1f}\\%}}}}}},
    fill={{rgb,255:red,{color_rgb.split(',')[0]};green,{color_rgb.split(',')[1]};blue,{color_rgb.split(',')[2]}}},
    circle,
    inner sep=1.5pt
] at (axis cs:{best_iter},{best_acc}) {{}};

"""
    
    latex_code += """\\end{axis}
\\end{tikzpicture}
\\caption{Accuracy comparison across different class weighting strategies during training. 
Class weights are shown in the legend as [Frail, Prefrail, Nonfrail]. 
The best accuracy point for each strategy is annotated.}
\\label{fig:part2_accuracy_comparison}
\\end{figure}
"""
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_code)
    
    print(f"✓ Saved standalone LaTeX file: {output_file}")
    return output_file

def main():
    csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    print("=" * 80)
    print("GENERATING LATEX CODE FOR PART 2 ACCURACY COMPARISON")
    print("=" * 80)
    
    experiments = load_part2_data(csv_file)
    
    if not experiments:
        print("\n❌ No Part 2 data found in CSV!")
        return
    
    print(f"\nFound {len(experiments)} Part 2 experiment(s)")
    
    # Generate standalone LaTeX (for including in Overleaf)
    standalone_file = generate_standalone_latex(experiments)
    
    # Generate complete document (for testing)
    complete_file = generate_latex_code(experiments)
    
    print("\n" + "=" * 80)
    print("✓ LaTeX files generated!")
    print("=" * 80)
    print("\nFor Overleaf:")
    print(f"  1. Upload: {standalone_file}")
    print("  2. In your main .tex file, add before \\begin{document}:")
    print("     \\usepackage{pgfplots}")
    print("     \\usepackage{times}")
    print("     \\pgfplotsset{compat=1.18}")
    print("  3. Include the figure with:")
    print(f"     \\input{{{os.path.basename(standalone_file)}}}")
    print("\nOr use the complete document:")
    print(f"  - Upload: {complete_file}")
    print("  - Compile directly in Overleaf")
    print("=" * 80)

if __name__ == '__main__':
    main()

