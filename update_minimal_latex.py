#!/usr/bin/env python3
"""Update the minimal LaTeX file to match the original Python graph exactly."""

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
                })
    
    # Sort by iteration and remove duplicates
    for exp_name in experiments:
        experiments[exp_name].sort(key=lambda x: x['iteration'])
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
    """Get class weights for an experiment (train-only split weights)."""
    weights_map = {
        'REDO_insqrt': [1.130, 0.935, 0.935],
        'REDO_balnormal': [1.267, 0.867, 0.867],
        'REDO_log': [1.138, 0.931, 0.931],
        'REDO_smooth': [0.778, 1.111, 1.111],
        'REDO_uniform': [1.000, 1.000, 1.000]
    }
    return weights_map.get(exp_name, [1.0, 1.0, 1.0])

def get_legend_label(exp_name):
    """Get legend label with class weights."""
    clean_name = get_clean_experiment_name(exp_name)
    weights = get_class_weights(exp_name)
    weights_str = f"[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
    return f"{clean_name} {weights_str}"

def main():
    csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    experiments = load_part2_data(csv_file)
    
    # Find max iteration
    max_iter = 0
    for data in experiments.values():
        if data:
            max_iter = max(max_iter, max(d['iteration'] for d in data))
    
    # Tab10 colormap RGB values (0-255) - exact match to matplotlib tab10 with linspace(0, 1, 5)
    # Python uses: colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    # This gives interpolated colors, not discrete ones!
    tab10_colors_rgb = [
        (31, 119, 180),   # Blue - Balanced Normal (tab10[0.0])
        (44, 160, 44),    # Green - Inverse Sqrt (tab10[0.25])
        (140, 86, 75),    # Brown - Log (tab10[0.5])
        (127, 127, 127),  # Gray - Smooth (tab10[0.75])
        (23, 190, 207),   # Cyan - Uniform (tab10[1.0])
    ]
    
    # Marker styles: ['o', 's', '^', 'D', 'v']
    markers = ['o', 'square', 'triangle', 'diamond', 'triangle*']
    
    # Line styles: ['-', '--', '-.', ':', '-']
    line_styles = ['solid', 'dashed', 'dashdotted', 'dotted', 'solid']
    
    # Experiment order (sorted)
    exp_order = sorted(experiments.keys())
    
    # Set x-axis ticks
    if max_iter <= 15000:
        x_ticks = list(range(0, int(max_iter) + 1, 2000))
    else:
        x_ticks = list(range(0, int(max_iter) + 1, 5000))
    
    latex_code = f"""% Part 2: Class Weights - Accuracy Comparison (Minimal PGFPlots)
% Matches the original Python graph exactly
% Requires: \\usepackage{{pgfplots}}, \\usepackage{{times}}, \\usepackage{{float}}, \\pgfplotsset{{compat=1.18}}

\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=14cm,
    height=8cm,
    xlabel={{Iteration}},
    ylabel={{Accuracy (\\%)}},
    title={{Part 2: Class Weights - Accuracy Comparison Over Training}},
    xmin=0,
    xmax={int(max_iter * 1.02)},
    ymin=0,
    ymax=100,
    grid=major,
    grid style={{dashed, gray!30, line width=0.8pt, opacity=0.25}},
    legend style={{
        at={{(0.98,0.02)}},
        anchor=south east,
        cells={{anchor=west}},
        font=\\small,
        fill=white,
        fill opacity=0.95,
        draw=gray,
        line width=1.2pt,
        rounded corners,
        shadow
    }},
    tick label style={{font=\\small}},
    label style={{font=\\bfseries}},
    title style={{font=\\bfseries}},
    axis background/.style={{fill=gray!2}},
    every axis plot/.append style={{line width=2.8pt, opacity=0.85}},
    ytick={{0,10,20,30,40,50,60,70,80,90,100}},
    xtick={{{','.join(map(str, x_ticks))}}},
    scaled ticks=false
]

"""
    
    # Track best points for annotations
    best_points = {}
    
    for i, exp_name in enumerate(exp_order):
        data = experiments[exp_name]
        if not data:
            continue
        
        # Get color (tab10 colormap)
        r, g, b = tab10_colors_rgb[i % len(tab10_colors_rgb)]
        color_def = f"{{rgb,255:red,{r};green,{g};blue,{b}}}"
        
        # Get marker and line style
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        
        # Get legend label
        legend_label = get_legend_label(exp_name).replace('_', '\\_')
        
        # Find best accuracy point
        best_idx = max(range(len(data)), key=lambda j: data[j]['accuracy'])
        best_acc = data[best_idx]['accuracy']
        best_iter = data[best_idx]['iteration']
        best_points[exp_name] = (best_iter, best_acc, color_def)
        
        # Create coordinate string
        coords = ' '.join([f"({d['iteration']},{d['accuracy']})" for d in data])
        
        # Build plot options matching Python exactly
        # markersize=6 in matplotlib ≈ 4pt in pgfplots
        plot_opts = [
            f"color={color_def}",
            f"mark={marker}",
            f"mark options={{fill=white, draw={color_def}, line width=1.5pt, mark size=4pt}}",
            f"style={line_style}",
            "opacity=0.85"
        ]
        
        plot_opts_str = ',\n    '.join(plot_opts)
        
        latex_code += f"""\\addplot[
    {plot_opts_str}
] coordinates {{
    {coords}
}};
\\addlegendentry{{{legend_label}}}

"""
    
    # Add best point annotations (matching Python: fontsize=9, bold, with bbox)
    for exp_name, (best_iter, best_acc, color_def) in best_points.items():
        latex_code += f"""\\node[
    pin={{[pin distance=5pt, pin edge={{->, {color_def}, line width=1.5pt, opacity=0.7}}]above right:{{\\textbf{{\\small {best_acc:.1f}\\%}}}}}},
    fill={color_def},
    circle,
    inner sep=1.5pt,
    opacity=0.85
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
    
    output_file = 'results_visualization/part2_accuracy_minimal.tex'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_code)
    
    print(f"✓ Updated minimal LaTeX file: {output_file}")
    print(f"  - Tab10 colors (exact match)")
    print(f"  - Markers: o, square, triangle, diamond, triangle*")
    print(f"  - Line styles: solid, dashed, dashdotted, dotted, solid")
    print(f"  - Line width: 2.8pt")
    print(f"  - Marker size: 2.5pt with white fill")
    print(f"  - Best point annotations included")
    print(f"  - Y-axis ticks: every 10")
    print(f"  - X-axis ticks: every 2000")

if __name__ == '__main__':
    main()

