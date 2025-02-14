import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score
import matplotlib.pyplot as plt
import time
start_time = time.time()

# Load dataset
input_file = "combined_top1.csv"
combined_df_top_1 = pd.read_csv(input_file)

# Define scoring parameters
scoring_parameters = [
    'Conf. Score', 'Average pLDDT', 'Average PAE (L->P)', 'Average PAE (P->L)', 'Dimer PAE', '1client_Conf. Score', '1client_Average pLDDT', '1client_Average PAE (L->P)', '1client_Average PAE (P->L)', '1client_Dimer PAE'
]

# Filter dataset to only include "binder" and "nonbinder"
filtered_df = combined_df_top_1[combined_df_top_1['BindingStatus'].isin(['binder', 'nonbinder'])]

# Convert "binder" to 1 and "nonbinder" to 0
y = (filtered_df['BindingStatus'] == 'binder').astype(int)

# Update X to match the filtered rows
X = filtered_df[scoring_parameters]

# Define precision for different parameters
precision_mapping = {
    'Conf. Score': 2, 'Average pLDDT': 3, 'Average PAE (L->P)': 3, 'Average PAE (P->L)': 3, 'Dimer PAE': 4, '1client_Conf. Score': 2, '1client_Average pLDDT': 3, '1client_Average PAE (L->P)': 3, '1client_Average PAE (P->L)': 3, '1client_Dimer PAE': 4
}

# Store TPR and FPR values separately for each run
roc_points = {}

# Function to update ROC data structure
def update_roc(fpr, tpr, thresholds, fraction, iteration):
    key = (fraction, iteration)
    if key not in roc_points:
        roc_points[key] = {}
    for i in range(len(fpr)):
        fpr_value = round(fpr[i], 2)
        if fpr_value not in roc_points[key] or tpr[i] > roc_points[key][fpr_value][0]:
            roc_points[key][fpr_value] = (tpr[i], thresholds)

# Function to apply thresholds
def apply_thresholds(X, thresholds):
    y_pred = np.ones(X.shape[0], dtype=bool)
    for i, param in enumerate(scoring_parameters):
        if 'Conf. Score' in param or 'pLDDT' in param:
            y_pred &= X[param] >= thresholds[i]  # Higher is better
        else:
            y_pred &= X[param] <= thresholds[i]  # Lower is better
    
    return y_pred.astype(int)

# Genetic Algorithm with Differential Evolution
def ga_objective(thresholds, X_train, y_train, fraction, iteration):
    y_pred = apply_thresholds(X_train, thresholds)
    fpr, tpr, _ = roc_curve(y_train, y_pred)
    key = (fraction, iteration)
    update_roc(fpr, tpr, thresholds, fraction, iteration)
    cumulative_auroc = auc(*zip(*[(fpr, tpr) for fpr, (tpr, _) in sorted(roc_points[key].items())]))
    return -cumulative_auroc  # Minimize negative cumulative AUROC

#Define sample fractions
sample_fractions = np.arange(0.1,1,0.1)

#Results storage
results = []

for fraction in sample_fractions:
    fraction_results = []
    fraction_train_results = []
    for i in range(10):
        train_indices = np.random.choice(X.index, size=int(fraction * len(X)), replace=False)
        test_indices = list(set(X.index) - set(train_indices))

        X_train, y_train = X.loc[train_indices], y.loc[train_indices]
        X_test, y_test = X.loc[test_indices], y.loc[test_indices]

        # Define bounds based on training set
        bounds = [(round(X_train[param].min(), precision_mapping[param]), 
                   round(X_train[param].max(), precision_mapping[param])) for param in scoring_parameters]
        result = differential_evolution(ga_objective, bounds, args=(X_train, y_train, fraction, i), updating='deferred',popsize=10000)
        best_ga_thresholds = [round(val, precision_mapping[param]) for val, param in zip(result.x, scoring_parameters)]

        ga_y_pred = apply_thresholds(X, best_ga_thresholds)
        ga_fpr, ga_tpr, _ = roc_curve(y, ga_y_pred)
        update_roc(ga_fpr, ga_tpr, best_ga_thresholds, fraction, i)
        ga_auroc = auc(*zip(*[(fpr, tpr) for fpr, (tpr, _) in sorted(roc_points[(fraction, i)].items())]))

        # Select best threshold from training based on FPR * (1 - TPR)
        best_threshold = max(roc_points[(fraction, i)].items(), key=lambda x: accuracy_score(y_train, apply_thresholds(X_train, x[1][1])))[1][1]

        # Evaluate on training set
        y_train_pred = apply_thresholds(X_train, best_threshold)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Evaluate on test set
        y_test_pred = apply_thresholds(X_test, best_threshold)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        fraction_results.append(test_accuracy)
        fraction_train_results.append(train_accuracy)

    results.append((fraction, fraction_results, fraction_train_results))

# Print results
print("\nTraining Fraction | Mean Test Accuracy | Std Test Accuracy | Mean Train Accuracy | Std Train Accuracy")
for fraction, test_acc_list, train_acc_list in results:
    print(f"{fraction:.1f} | {np.mean(test_acc_list):.4f} | {np.std(test_acc_list):.4f} | {np.mean(train_acc_list):.4f} | {np.std(train_acc_list):.4f}")

print("--- %s seconds ---" % (time.time() - start_time))

# Plot Mean Accuracy per Fraction
plt.figure(figsize=(8, 6))
fractions = [fraction for fraction, _, _ in results]
mean_test_accuracies = [np.mean(test_acc_list) for _, test_acc_list, _ in results]
std_test_accuracies = [np.std(test_acc_list) for _, test_acc_list, _ in results]
mean_train_accuracies = [np.mean(train_acc_list) for _, _, train_acc_list in results]
std_train_accuracies = [np.std(train_acc_list) for _, _, train_acc_list in results]

# Plot test accuracy
plt.errorbar(fractions, mean_test_accuracies, yerr=std_test_accuracies, fmt='-o', capsize=5, label="Test Accuracy", color='blue')

# Plot training accuracy
plt.errorbar(fractions, mean_train_accuracies, yerr=std_train_accuracies, fmt='-s', capsize=5, label="Training Accuracy", color='red')

plt.xlabel("Training Fraction")
plt.ylabel("Mean Accuracy")
plt.title("Mean Training & Test Accuracy for Different Training Fractions")
plt.legend()
plt.show()

#get iteration from the 0.9 fraction training with the larget auroc by evaluating the total auroc for roc_points[(0.9,i)]
fraction_key = 0.9
best_iteration = max(range(10), key=lambda i: auc(*zip(*[(fpr, tpr) for fpr, (tpr, _) in sorted(roc_points[(fraction_key, i)].items())])))


# Find thresholds for FPR = 0.08
thresholds_fpr_08 = roc_points[(fraction_key,best_iteration)].get(0.08, ("Not found", "Not found"))

# Find thresholds for TPR = 0.92 (closest match)
thresholds_tpr_92 = min(roc_points[(fraction_key,best_iteration)].items(), key=lambda x: abs(x[1][0] - 0.92))

# Find thresholds that maximizes accuracy
accuracy_thresholds = max(roc_points[(fraction_key,best_iteration)].items(), key=lambda x: accuracy_score(y, apply_thresholds(X, x[1][1])))

# Find thresholds that maximizes precision
precision_thresholds = max(roc_points[(fraction_key,best_iteration)].items(), key=lambda x: precision_score(y, apply_thresholds(X, x[1][1])))

# Print results
print("\nThresholds for FPR = 0.08:")
print("TPR:", thresholds_fpr_08[0], "Thresholds:", thresholds_fpr_08[1])

print("\nThresholds for TPR ≈ 0.92:")
print("FPR:", thresholds_tpr_92[0],  "TPR:", thresholds_tpr_92[1][0],  "Thresholds:", thresholds_tpr_92[1][1])

print("\nThresholds that maximize accuracy:")
print("FPR:", accuracy_thresholds[0], "TPR:", accuracy_thresholds[1][0], "Thresholds:", accuracy_thresholds[1][1])

print("\nThresholds that maximize precision:")
print("FPR:", precision_thresholds[0], "TPR:", precision_thresholds[1][0], "Thresholds:", precision_thresholds[1][1])

#plot the ROC curve for the 0.9 fraction iteration that achieves the highest AUROC
# Extract the best ROC data for plotting
best_fpr_tpr = sorted(roc_points[(fraction_key, best_iteration)].items())
best_fpr = [fpr for fpr, _ in best_fpr_tpr]
best_tpr = [tpr for _, (tpr, _) in best_fpr_tpr]
best_auroc = auc(best_fpr, best_tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(best_fpr, best_tpr, marker='o', linestyle='-', label=f'ROC (AUROC = {best_auroc:.3f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title(f"ROC Curve for Training Fraction {fraction_key} (Iteration {best_iteration})")
plt.legend()
plt.grid()
plt.show()

# Print the iteration with the best AUROC
print(f"\nBest AUROC for training fraction {fraction_key:.1f} achieved at iteration {best_iteration} with AUROC = {best_auroc:.4f}")