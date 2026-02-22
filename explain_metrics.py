#!/usr/bin/env python3
"""
Explanation of F1, Precision, and Recall metrics for frailty classification.
"""

print("=" * 80)
print("UNDERSTANDING CLASSIFICATION METRICS")
print("=" * 80)

print("""
For your frailty classification task (3 classes: Frail, Prefrail, Nonfrail),
here's what each metric means:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PRECISION (Positive Predictive Value)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Precision = True Positives / (True Positives + False Positives)

"What percentage of my predictions for this class were correct?"

Example for "Frail" class:
  - You predicted 10 subjects as "Frail"
  - 8 of them were actually "Frail" (True Positives)
  - 2 of them were actually "Prefrail" or "Nonfrail" (False Positives)
  - Precision = 8 / (8 + 2) = 0.80 = 80%

High Precision = When you predict a class, you're usually right
Low Precision = You're making many false alarms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. RECALL (Sensitivity, True Positive Rate)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recall = True Positives / (True Positives + False Negatives)

"What percentage of actual cases of this class did I find?"

Example for "Frail" class:
  - There are 15 actual "Frail" subjects in your test set
  - You correctly identified 12 of them (True Positives)
  - You missed 3 of them (False Negatives - predicted as Prefrail/Nonfrail)
  - Recall = 12 / (12 + 3) = 0.80 = 80%

High Recall = You're finding most of the actual cases
Low Recall = You're missing many actual cases

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. F1 SCORE (Harmonic Mean of Precision and Recall)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

F1 = 2 × (Precision × Recall) / (Precision + Recall)

A single number that balances both Precision and Recall.

Example:
  - Precision = 0.80 (80%)
  - Recall = 0.80 (80%)
  - F1 = 2 × (0.80 × 0.80) / (0.80 + 0.80) = 0.80 = 80%

High F1 = Good balance between precision and recall
Low F1 = Either precision or recall (or both) is low

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. SPECIFICITY (True Negative Rate)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Specificity = True Negatives / (True Negatives + False Positives)

"What percentage of non-cases did I correctly identify as non-cases?"

Example for "Frail" class:
  - There are 30 subjects who are NOT "Frail" (they're Prefrail or Nonfrail)
  - You correctly identified 27 of them as NOT "Frail" (True Negatives)
  - You incorrectly predicted 3 of them as "Frail" (False Positives)
  - Specificity = 27 / (27 + 3) = 0.90 = 90%

High Specificity = You're good at ruling out the class
Low Specificity = You're making many false alarms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFUSION MATRIX EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Let's say you have a confusion matrix:

                    Predicted
                 Frail  Prefrail  Nonfrail
Actual Frail       8       2        0      (10 total Frail)
      Prefrail     1       6        3      (10 total Prefrail)
      Nonfrail     0       2        8      (10 total Nonfrail)

For "Frail" class:
  - True Positives (TP) = 8 (correctly predicted as Frail)
  - False Positives (FP) = 1 + 0 = 1 (predicted as Frail but actually not)
  - False Negatives (FN) = 2 + 0 = 2 (actually Frail but predicted as other)
  - True Negatives (TN) = 6 + 3 + 2 + 8 = 19 (correctly predicted as not Frail)

  Precision = TP / (TP + FP) = 8 / (8 + 1) = 8/9 = 88.89%
  Recall = TP / (TP + FN) = 8 / (8 + 2) = 8/10 = 80.00%
  Specificity = TN / (TN + FP) = 19 / (19 + 1) = 19/20 = 95.00%
  F1 = 2 × (0.8889 × 0.8000) / (0.8889 + 0.8000) = 0.8421 = 84.21%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MACRO-AVERAGED vs PER-CLASS METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In your TensorBoard files, you see:

1. MACRO-AVERAGED metrics (what you currently have):
   - test_precision/ = Average precision across all 3 classes
   - test_recall/ = Average recall across all 3 classes
   - test_f1/ = Average F1 across all 3 classes
   
   Example:
     Frail Precision: 80%
     Prefrail Precision: 70%
     Nonfrail Precision: 90%
     Macro-Averaged Precision = (80 + 70 + 90) / 3 = 80%

2. PER-CLASS metrics (what you'll get after re-running evaluation):
   - test_precision/Frail = Precision for Frail class only
   - test_precision/Prefrail = Precision for Prefrail class only
   - test_precision/Nonfrail = Precision for Nonfrail class only
   - Same for recall, specificity, and F1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THESE METRICS MATTER FOR FRAILTY CLASSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For medical/health applications like frailty classification:

HIGH PRECISION is important when:
  - You want to minimize false alarms
  - Example: If you predict someone is "Frail", you want to be confident

HIGH RECALL is important when:
  - You want to catch all actual cases
  - Example: You don't want to miss someone who is actually "Frail"

HIGH F1 is important when:
  - You want a balanced model that's both precise and comprehensive
  - Good general performance indicator

HIGH SPECIFICITY is important when:
  - You want to correctly identify people who are NOT frail
  - Helps avoid unnecessary interventions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR CURRENT METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

From your REDO experiments, you're seeing:
- Accuracy: Overall percentage of correct predictions
- Precision (macro): Average precision across all classes
- Recall (macro): Average recall across all classes (same as Sensitivity)
- F1 (macro): Average F1 across all classes

Example from REDO_insqrt at iteration 10000:
  - Accuracy: 73.33% (11 out of 15 test subjects correctly classified)
  - Precision: 0.7738 (77.38%) - When you predict a class, you're right ~77% of the time
  - Recall: 0.7389 (73.89%) - You're finding ~74% of actual cases
  - F1: 0.7389 (73.89%) - Balanced performance

After re-running evaluation, you'll also get:
  - Per-class Precision, Recall, Specificity, and F1
  - This tells you which classes your model is best/worst at predicting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("=" * 80)
print("For more details, check your confusion matrices in the evaluation output!")
print("=" * 80)

