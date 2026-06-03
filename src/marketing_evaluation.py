import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

def plot_pr_curve(y_true, y_probs):
    """Plots the Precision-Recall curve to visualize the tradeoff."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall[:-1], precision[:-1], label='LightGBM', color='#2ecc71', linewidth=2)
    plt.title('Precision-Recall Tradeoff', fontsize=14, pad=15)
    plt.xlabel('Recall (Percentage of Real Buyers Caught)', fontsize=12)
    plt.ylabel('Precision (Percentage of Triggers that were Correct)', fontsize=12)
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_threshold(y_true, y_probs, threshold=0.5):
    """Simulates business performance at a specific probability threshold."""
    # Convert probabilities to 1 or 0 based on our custom threshold
    y_pred_custom = (y_probs >= threshold).astype(int)
    
    print(f"📊 Business Simulation at Threshold: {threshold}")
    print("-" * 40)
    print(classification_report(y_true, y_pred_custom))
    
    # Generate a quick text-based confusion matrix summary
    cm = confusion_matrix(y_true, y_pred_custom)
    print(f"Total True Buyers Caught: {cm[1, 1]}")
    print(f"True Buyers Missed (Slipped through): {cm[1, 0]}")
    print(f"False Positives (Money wasted on window shoppers): {cm[0, 1]}")