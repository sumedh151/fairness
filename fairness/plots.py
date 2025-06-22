import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_lines(x_data, *y_data_arrays, labels=None, colors=None, title='Plot', xlabel='X-axis', ylabel='Y-axis'):
    """
    Plot multiple lines on the same graph.

    This function creates a line plot with multiple lines, where each line represents a dataset passed as an argument. It supports customizable colors and labels for each line and includes options to set the plot's title and axis labels.

    Parameters:
    - x_data (array-like): The x-axis values common to all lines. For example, this could be epochs, time points, or any sequence of x-values.
    - *y_data_arrays (array-like): Variable length argument list where each argument is an array of y-axis values corresponding to a line on the plot. Each array should have the same length as `x_data`.
    - labels (list of str, optional): A list of labels for each line. If provided, these labels will appear in the plot legend. The length of this list should match the number of `y_data_arrays`. Default is None.
    - colors (list of str, optional): A list of colors for each line. Colors should be specified as valid matplotlib color strings. If not provided, colors will be generated using the `viridis` colormap. The length of this list should match the number of `y_data_arrays`. Default is None.
    - title (str, optional): The title of the plot. Default is 'Plot'.
    - xlabel (str, optional): The label for the x-axis. Default is 'X-axis'.
    - ylabel (str, optional): The label for the y-axis. Default is 'Y-axis'.

    Example:
    >>> epochs_total = [0, 1, 2, 3, 4]
    >>> gradient_norms_advA = [0.1, 0.2, 0.15, 0.25, 0.3]
    >>> gradient_norms_enc = [0.05, 0.1, 0.1, 0.2, 0.15]
    >>> gradient_norms_pred = [0.2, 0.3, 0.25, 0.4, 0.35]
    >>> labels = ['Gradient Norms AdvA', 'Gradient Norms Enc', 'Gradient Norms Pred']
    >>> plot_lines(epochs_total, gradient_norms_advA, gradient_norms_enc, gradient_norms_pred, labels=labels, title='Gradient Norms Over Epochs', xlabel='Epoch', ylabel='Average Gradient Norm')

    Notes:
    - If the number of lines exceeds the number of colors provided, the function will use a default color for additional lines.
    - The `viridis` colormap is used to generate colors if `colors` is not specified, with the number of colors matching the number of `y_data_arrays`.
    """

    if colors is None:
        colors = cm.viridis(np.linspace(0, 1, len(y_data_arrays)))
    
    plt.figure(figsize=(10, 6))
    for i, y_data in enumerate(y_data_arrays):
        color = colors[i] if i < len(colors) else 'C0'  # Fallback color
        label = labels[i] if labels is not None and i < len(labels) else None
        plt.plot(x_data, y_data, color=color, marker='o', label=label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.grid(True)
    if labels:
        plt.legend()
    
    plt.show()
    

def plot_comparison_heatmaps(df1, df2, figsize=(20, 8), cmap='coolwarm', vmin=-1, vmax=1):
    """
    Plot side-by-side heatmaps for the correlation matrices of two DataFrames.

    Parameters:
    - df1 (DataFrame): The first DataFrame for which the correlation matrix will be computed and plotted.
    - df2 (DataFrame): The second DataFrame for which the correlation matrix will be computed and plotted.
    - figsize (tuple, optional): The size of the figure (width, height). Default is (20, 8).
    - cmap (str, optional): The colormap to use for the heatmaps. Default is 'coolwarm'.
    - vmin (float, optional): The minimum value for the colormap scale. Default is -1.
    - vmax (float, optional): The maximum value for the colormap scale. Default is 1.
    """
    
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    sns.heatmap(df1.corr(), annot=True, cmap=cmap, fmt='.4f', vmin=vmin, vmax=vmax)
    plt.title('Correlation Matrix - Before')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(df2.corr(), annot=True, cmap=cmap, fmt='.4f', vmin=vmin, vmax=vmax)
    plt.title('Correlation Matrix - After')
    
    plt.suptitle('Pearson Correlation Heatmap', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    

def plot_auc_roc_curve(y_true, y_logits, title='ROC Curve'):
    """
    Plot the ROC curve and calculate the AUC score.

    Parameters:
    y_true (array-like): True binary labels (0 or 1) for the samples.
    y_logits (array-like): Predicted probabilities for the positive class.
    title (str, optional): Title of the plot. Default is 'ROC Curve'.

    Returns:
    None: The function plots the ROC curve and displays the AUC score.
    """
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_logits)
    
    # Compute AUC score
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()
