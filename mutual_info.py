import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

# Step 1: Read in the dataset (unfiltered)
data = pd.read_csv('2client_results.csv')
data = pd.read_csv('1client_results.csv')
data = pd.read_csv('success_rate_results.csv')

# Step 2: Define filter functions for sub-datasets

# Condition 1: Column 1 must have 'TQT' at positions 5-7
def has_tqt_at_position(sequence):
    return sequence[4:7] == 'TQT'  # Python uses zero-based indexing

# Condition 2: Column 3 must have a length not equal to 12 or 14
def length_not_12_or_14(entry):
    return len(entry) != 14 and len(entry) != 12

# Step 3: Bin the affinity values
affinity_bins = [0, .316, 3.16, 31.6, 100]
affinity_labels = ['0-.316', '.316-3.16', '3.16-31.6', '31.6-100']

# Step 4: Get the global min/max from the full dataset for each score column
score_columns = data.iloc[:, 3:13].columns  # Columns 4-13 for scoring metrics
global_score_ranges = {}
quantile_edges = {}
num_bins = 6

for column in score_columns:
    score_range_min = data[column].min()  # Full dataset min
    score_range_max = data[column].max()  # Full dataset max
    quantile_edges[column] = np.quantile(data[column], np.linspace(0, 1, num_bins))
    
    # Store the global range for later use
    global_score_ranges[column] = (score_range_min, score_range_max)

# Step 5: Now apply the filter and analyze each sub-dataset
filtered_data = data[data.iloc[:, 0].apply(has_tqt_at_position) & data.iloc[:, 2].apply(length_not_12_or_14)].copy()
print("Dataset contains",filtered_data.shape[0],"entries.")

# Step 6: Bin the affinities (column 2) into the defined ranges
filtered_data.loc[:, 'Binned_Affinity'] = pd.cut(filtered_data.iloc[:, 1], bins=affinity_bins, labels=affinity_labels, include_lowest=True)

# Step 7: Discretize the score columns using geomspace based on the global score ranges and calculate AMI for each score column
ami_results = {}

for column in score_columns:
    # Use geomspace to create 5 bins based on the full dataset range for each score
    score_range_min, score_range_max = global_score_ranges[column]
    
    if score_range_min > 0:  # Geomspace can't handle non-positive numbers
        #score_bins = np.geomspace(score_range_min, score_range_max, 6)  # 5 intervals, 6 bin edges
        score_bins = np.linspace(score_range_min, score_range_max, 6)  # 5 intervals, 6 bin edges
    else:
        # Fallback to linspace if the score contains non-positive values
        score_bins = np.linspace(score_range_min, score_range_max, 6)
    
    score_discretized = np.digitize(filtered_data[column], bins=score_bins)
    #score_discretized = np.digitize(filtered_data[column], bins=quantile_edges[column])
    if column == "blah":
        #print(filtered_data['Binned_Affinity'])
        #print(quantile_edges[column])
        #print(filtered_data[column])
        print(score_bins)
        print(score_discretized)
        print("1s:", np.count_nonzero(score_discretized==1))
        print("2s:", np.count_nonzero(score_discretized==2))
        print("3s:", np.count_nonzero(score_discretized==3))
        print("4s:", np.count_nonzero(score_discretized==4))
        print("5s:", np.count_nonzero(score_discretized==5))
        print("6s:", np.count_nonzero(score_discretized==6))
    
    # Calculate AMI between binned affinities and the current score column
    ami = adjusted_mutual_info_score(filtered_data['Binned_Affinity'], score_discretized)
    ami_results[column] = ami
    print(f"AMI for {column}: {ami}")

# Step 8: Output the results (save to CSV file)
ami_df = pd.DataFrame(list(ami_results.items()), columns=['Metric', 'Adjusted Mutual Information'])
ami_df.to_csv('adjusted_mutual_information_results_filtered.csv', index=False)

print("AMI calculation complete!")
