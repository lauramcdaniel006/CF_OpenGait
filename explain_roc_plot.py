#!/usr/bin/env python3
"""
Explain where ROC and AUC are in the plot.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

# Load the data
prob_file = 'output/roc_probabilities/probs_iter_final.npz'
data = np.load(prob_file)
probs = data['probs']
true_ids = data['true_ids']
class_names = data['class_names']

print("="*70)
print("WHERE IS THE ROC AND AUC?")
print("="*70)
print()
print("1. ROC (Receiver Operating Characteristic) CURVE:")
print("   → The ROC curve is the COLORED LINE itself")
print("   → Each colored line plots: TPR (y-axis) vs FPR (x-axis)")
print("   → There are 3 ROC curves (one for each class)")
print()
print("2. AUC (Area Under the Curve):")
print("   → AUC is the AREA under each ROC curve")
print("   → It's shown in the LEGEND as 'Class Name (AUC = X.XXX)'")
print("   → Higher AUC = better performance")
print()

# Compute AUC for each class
print("Current AUC values:")
print("-"*70)
for i in range(len(class_names)):
    y_binary = (true_ids == i).astype(int)
    y_score = probs[:, i]
    fpr, tpr, thresholds = roc_curve(y_binary, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"  {class_names[i]}: AUC = {roc_auc:.4f}")
print()

print("="*70)
print("VISUAL EXPLANATION:")
print("="*70)
print()
print("In your plot (roc_curves_*.png):")
print()
print("  ┌─────────────────────────────────────────┐")
print("  │  Model Name - ROC Curves                │")
print("  │                                         │")
print("  │    1.0 ┤                                 │")
print("  │        │     ╱─── ROC Curve (colored line)│")
print("  │    0.8 ┤   ╱                            │")
print("  │  TPR   │  ╱                             │")
print("  │    0.6 ┤ ╱                              │")
print("  │        │╱                               │")
print("  │    0.4 ┤                                │")
print("  │        │                                │")
print("  │    0.2 ┤                                │")
print("  │        │                                │")
print("  │    0.0 └─────────────────────────────── │")
print("  │        0.0  0.2  0.4  0.6  0.8  1.0    │")
print("  │              FPR (False Positive Rate)  │")
print("  │                                         │")
print("  │  Legend (bottom right):                 │")
print("  │  ┌──────────────────────────────┐       │")
print("  │  │ Frail (AUC = 0.750)  ←─── AUC│       │")
print("  │  │ Prefrail (AUC = 0.740) ←─── AUC│     │")
print("  │  │ Nonfrail (AUC = 0.852) ←─── AUC│     │")
print("  │  └──────────────────────────────┘       │")
print("  └─────────────────────────────────────────┘")
print()
print("KEY POINTS:")
print("  • The COLORED LINES = ROC curves")
print("  • The NUMBER in the legend (AUC = X.XXX) = AUC value")
print("  • The AREA under each line = AUC (calculated, not drawn)")
print()

