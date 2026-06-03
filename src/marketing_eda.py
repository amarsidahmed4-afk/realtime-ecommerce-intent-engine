# marketing_eda.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df):
    """Plots a correlation heatmap for numerical features."""
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        vmin=-1, 
        vmax=1, 
        linewidths=0.5
    )
    plt.title('Feature Correlation Matrix', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

def plot_feature_densities(df, features, target='Revenue'):
    """Plots KDE density curves for given features split by target class."""
    # Dynamically calculate rows needed for a 2-column layout
    num_features = len(features)
    cols = 2
    rows = (num_features + 1) // 2
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        sns.kdeplot(
            data=df, 
            x=col, 
            hue=target, 
            fill=True,
            common_norm=False,
            alpha=0.5,
            ax=axes[i]
        )
        axes[i].set_title(f'Density Distribution: {col}', fontsize=12)
        axes[i].set_xlabel('')
        
    # Hide any empty subplots if we pass an odd number of features
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()