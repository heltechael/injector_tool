import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
csv_file = 'Predicted_labels_rwm.csv'
df = pd.read_csv(csv_file)

unique_labels = df['pred_label_eppo'].unique()

# Compute the optimal threshold based on precision-recall curve
def compute_optimal_threshold(confidences, true_labels):
    precision, recall, thresholds = precision_recall_curve(true_labels, confidences)
    th_idx = len(precision) - next(i for i, x in enumerate(np.flip(precision)) if x < 1)
    if th_idx >= len(precision) - 1:
        th_idx = -1
    th_optim = thresholds[th_idx]
    return th_optim, precision[th_idx], recall[th_idx]

thresholds_list = []

for label in unique_labels:
    label_df = df[df['pred_label_eppo'] == label]
    true_labels = (label_df['EPPOcode'] == label).astype(int)
    
    # Compute the optimal threshold
    th_optim, precision_at_th, recall_at_th = compute_optimal_threshold(label_df['confidence_score'].values, true_labels.values)
    
    # Compute median threshold
    median_threshold = np.median(label_df['confidence_score'].values)
    
    # Compute mean and standard deviation of the confidence scores
    mu = np.mean(label_df['confidence_score'].values)
    sigma = np.std(label_df['confidence_score'].values)
    
    thresholds_list.append({
        'EPPO': label,
        'N_samples': len(label_df),
        'Threshold_optim': th_optim,
        'Threshold_median': median_threshold,
        'Precision': precision_at_th,
        'Recall': recall_at_th,
        'mu': mu,
        'sigma': sigma
    })

thresholds_df = pd.DataFrame(thresholds_list)

# Plot precision-recall curves and histograms for each label
fig, axs = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True, figsize=(15, 9))
axs = axs.flatten()
fig2, axs2 = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=False, figsize=(15, 9))
axs2 = axs2.flatten()

for i, label in enumerate(unique_labels):
    label_df = df[df['pred_label_eppo'] == label]
    true_labels = (label_df['EPPOcode'] == label).astype(int)
    precision, recall, thresholds = precision_recall_curve(true_labels, label_df['confidence_score'].values)
    th_idx = len(precision) - next(i for i, x in enumerate(np.flip(precision)) if x < 1)
    if th_idx >= len(precision) - 1:
        th_idx = -1

    axs[i].plot(recall, precision)
    axs[i].plot(recall[th_idx], precision[th_idx], 'x', label='th=' + str(thresholds[th_idx]))
    axs[i].set_title(label + '(N=' + str(len(label_df)) + ')')
    axs[i].set_xlabel('Recall')
    axs[i].set_ylabel('Precision')
    axs[i].legend(loc='lower left')

    # Plot histogram
    n, bins, patches = axs2[i].hist(label_df['confidence_score'].values, bins=30, density=True)
    mu = np.mean(label_df['confidence_score'].values)
    sigma = np.std(label_df['confidence_score'].values)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    axs2[i].plot(bins, y, '--')
    axs2[i].plot(thresholds_df[thresholds_df['EPPO'] == label]['Threshold_median'].values * np.ones((2,)), [0, 0.15], '-', label='th_median')
    axs2[i].plot(thresholds_df[thresholds_df['EPPO'] == label]['Threshold_optim'].values * np.ones((2,)), [0, 0.15], '-', label='th_optim')
    axs2[i].set_title(label + '(N=' + str(len(label_df)) + ')')
    axs2[i].legend(loc='upper right')

plt.show()

thresholds_df.to_csv('computed_thresholds.csv', index=False)
print("Thresholds computed and saved to 'computed_thresholds.csv'")
