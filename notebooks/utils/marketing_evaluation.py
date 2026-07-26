import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

def plot_pr_curve(y_true, y_probs, model_name="LightGBM"):
    """Plots the Precision-Recall curve to visualize the tradeoff."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap_score = average_precision_score(y_true, y_probs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall[:-1], 
        precision[:-1], 
        label=f'{model_name} (PR-AUC = {ap_score:.3f})', 
        color='#2ecc71', 
        linewidth=2
    )
    ax.set_title(f'Precision-Recall Tradeoff - {model_name}', fontsize=14, pad=15)
    ax.set_xlabel('Recall (Percentage of Real Buyers Caught)', fontsize=12)
    ax.set_ylabel('Precision (Percentage of Triggers that were Correct)', fontsize=12)
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    return fig, ax

def evaluate_threshold(y_true, y_probs, threshold=0.5):
    """Simulates business performance at a specific probability threshold."""
    y_pred_custom = (y_probs >= threshold).astype(int)
    
    print(f"📊 Business Simulation at Threshold: {threshold}")
    print("-" * 40)
    print(classification_report(y_true, y_pred_custom, zero_division=0))
    
    # Passing labels=[0, 1] ensures 2x2 matrix shape even at extreme thresholds
    cm = confusion_matrix(y_true, y_pred_custom, labels=[0, 1])
    print(f"Total True Buyers Caught: {cm[1, 1]}")
    print(f"True Buyers Missed (Slipped through): {cm[1, 0]}")
    print(f"False Positives (Money wasted on window shoppers): {cm[0, 1]}")
