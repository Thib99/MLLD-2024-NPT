
############################# Function to print an histogram of the model's data #############################
from matplotlib import pyplot as plt
import numpy as np


def printHistogram(title, full_data):
        
    # Model names
    model_names = [data['short_name'] for data in full_data.values()]

    # Mean F1 scores
    f1_scores = [data['f1'] for data in full_data.values()]

    # Mean false negatives
    false_negatives = [data['false_negatif'] for data in full_data.values()]

    # Width of the bars
    bar_width = 0.35

    # Position of the bars on the X-axis
    x = np.arange(len(model_names))

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Create bars for F1 on Y1 axis
    ax1.bar(x - bar_width/2, f1_scores, bar_width, color='green', label='F1 Score')

    # Y1 axis configuration for F1
    ax1.set_ylabel('F1 Score', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    # Create a second axis for false negatives on Y2 axis
    ax2 = ax1.twinx()
    ax2.bar(x + bar_width/2, false_negatives, bar_width, color='blue', alpha=0.5, label='False Negatives')

    # Y2 axis configuration for false negatives
    ax2.set_ylabel('False Negatives', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # X-axis configuration
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')

    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Title of the plot
    plt.title(title)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
    
    
################################ END ######################################################################