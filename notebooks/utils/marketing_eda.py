import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_correlation_matrix(df, mask_upper=True):
    """Plots a correlation heatmap for numerical features with optional upper mask."""
    corr_matrix = df.corr(numeric_only=True)
    
    # Optional upper-triangle mask to remove duplicate symmetrical values
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) if mask_upper else None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        vmin=-1, 
        vmax=1, 
        linewidths=0.5,
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_feature_densities(df, features, target='Revenue'):
    """Plots KDE density curves for given features split by target class."""
    if not features:
        print("⚠️ No features provided for density plotting.")
        return

    num_features = len(features)
    cols = 2
    rows = (num_features + 1) // 2
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    
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
        
    # Safely hide any unused empty subplots
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()
    return fig, axes
