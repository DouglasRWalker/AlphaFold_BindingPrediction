import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the cutoff values for success determination
cutoff = [0.7, 70.0, 4.0, 7.3, 1.15, 0.8, 65.1, 6.3, 9.7, 1.08]

def load_data(affinities_file, combined_file, linked_file):
    affinities_df = pd.read_csv(affinities_file, delimiter='\t')
    combined_df = pd.read_csv(combined_file)
    linked_df = pd.read_csv(linked_file)
    return affinities_df, combined_df, linked_df

def match_sequences(affinities_df, combined_df, linked_df):
    results_2client = []
    results_1client = []
    results_success_rate = []

    for _, affinity_row in affinities_df.iterrows():
        affinity_seq = affinity_row['Sequence']
        affinity_value = affinity_row['Affinity']

        # Filter combined_df for matching sequences (substring match)
        matching_combined_df = combined_df[combined_df['Full Sequence'].str.contains(affinity_seq, regex=False)]
        
        # Group by 'File Path' to only check each group once
        for file_path, group in matching_combined_df.groupby('File Path'):
            if group['BindingStatus'].iloc[0] == "binder":
                full_sequence = group['Full Sequence'].iloc[0]
                client_type = group['ClientType'].iloc[0]

                # Extract relevant scores
                best_conf = group['Conf. Score'].max()
                best_avg_plddt = group['Average pLDDT'].max()
                best_lp_pae = group['Average PAE (L->P)'].min()
                best_pl_pae = group['Average PAE (P->L)'].min()
                best_dimer_pae = group['Dimer PAE'].min()

                group_size = len(group)
                avg_conf = group['Conf. Score'].sum() / group_size
                avg_avg_plddt = group['Average pLDDT'].sum() / group_size
                avg_lp_pae = group['Average PAE (L->P)'].sum() / group_size
                avg_pl_pae = group['Average PAE (P->L)'].sum() / group_size
                avg_dimer_pae = group['Dimer PAE'].sum() / group_size

                if client_type == '2client':
                    results_2client.append([affinity_seq, affinity_value, full_sequence, best_conf, best_avg_plddt, best_lp_pae, best_pl_pae, best_dimer_pae, avg_conf, avg_avg_plddt, avg_lp_pae, avg_pl_pae, avg_dimer_pae])

                elif client_type == '1client':
                    results_1client.append([affinity_seq, affinity_value, full_sequence, best_conf, best_avg_plddt, best_lp_pae, best_pl_pae, best_dimer_pae, avg_conf, avg_avg_plddt, avg_lp_pae, avg_pl_pae, avg_dimer_pae])

        # Calculate success rate for linked_df
        matching_linked_df = linked_df[linked_df['Full Sequence'].str.contains(affinity_seq, regex=False)]

        # Group by 'File Path' to only check each group once
        for file_path, group in matching_linked_df.groupby('File Path'):
            if (group['BindingStatus'].iloc[0] == "binder") & (len(group) > 25):
                full_sequence = group['Full Sequence'].iloc[0]

                # Determine success for each entry
                num_successes = (
                    (group['Conf. Score'] >= cutoff[0]) & 
                    (group['Average pLDDT'] >= cutoff[1]) & 
                    (group['Average PAE (L->P)'] <= cutoff[2]) & 
                    (group['Average PAE (P->L)'] <= cutoff[3]) & 
                    (group['Dimer PAE'] <= cutoff[4]) &
                    (group['1client_Conf. Score'] >= cutoff[5]) & 
                    (group['1client_Average pLDDT'] >= cutoff[6]) & 
                    (group['1client_Average PAE (L->P)'] <= cutoff[7]) & 
                    (group['1client_Average PAE (P->L)'] <= cutoff[8]) & 
                    (group['1client_Dimer PAE'] <= cutoff[9])
                ).sum()

                success_rate = num_successes / len(group) if len(group) > 0 else np.nan
                #success_rate = (1-success_rate)*(1-success_rate)/success_rate if success_rate > 0 else 40
                #success_rate = (len(group)-num_successes)*(len(group)-num_successes)/num_successes if num_successes > 0 else (len(group)-num_successes)*(len(group)-num_successes)
                results_success_rate.append([affinity_seq, affinity_value, full_sequence, len(group), success_rate])

    # Convert results to DataFrames
    df_2client = pd.DataFrame(results_2client, columns=['Sequence', 'Affinity', 'Full Sequence', 'Best Conf. score', 'Best Average pLDDT', 'Best L->P PAE', 'Best P->L PAE', 'Best Dimer PAE', 'Avg Conf. score', 'Avg Average pLDDT', 'Avg L->P PAE', 'Avg P->L PAE', 'Avg Dimer PAE'])
    df_1client = pd.DataFrame(results_1client, columns=['Sequence', 'Affinity', 'Full Sequence', 'Best Conf. score', 'Best Average pLDDT', 'Best L->P PAE', 'Best P->L PAE', 'Best Dimer PAE', 'Avg Conf. score', 'Avg Average pLDDT', 'Avg L->P PAE', 'Avg P->L PAE', 'Avg Dimer PAE'])
    df_success_rate = pd.DataFrame(results_success_rate, columns=['Sequence', 'Affinity', 'Full Sequence', 'Reps', 'Success Rate'])

    return df_2client, df_1client, df_success_rate

def save_dataframes(df_2client, df_1client, df_success_rate):
    df_2client.to_csv('2client_results.csv', index=False)
    df_1client.to_csv('1client_results.csv', index=False)
    df_success_rate.to_csv('success_rate_results.csv', index=False)

def plot_results(df_2client, df_1client, df_success_rate):
    # Plotting code for scatterplots
    parameters = ['Best Conf. score', 'Best Average pLDDT', 'Best L->P PAE', 'Best P->L PAE', 'Best Dimer PAE', 'Avg Conf. score', 'Avg Average pLDDT', 'Avg L->P PAE', 'Avg P->L PAE', 'Avg Dimer PAE']

    for param in parameters:
        plt.figure()
        plt.scatter(df_2client['Affinity'], df_2client[param], label='2client')
        plt.scatter(df_1client['Affinity'], df_1client[param], label='1client', alpha=0.7)
        plt.xlabel('Affinity')
        plt.ylabel(param)
        plt.title(f'Affinity vs {param}')
        plt.legend()
        plt.show()

    # Plot success rate
    plt.figure()
    plt.scatter(df_success_rate['Affinity'], df_success_rate['Success Rate'])
    plt.xlabel('Affinity')
    plt.ylabel('Success Rate')
    plt.title('Affinity vs Success Rate')
    plt.show()

def calculate_r2_values(df, target_column):
    # Function to calculate R^2 values between Affinity and Success Rate
    def calculate_r2(x, y):
        if len(x) < 2 or len(y) < 2:
            return float('nan')  # Return NaN if R^2 calculation isn't possible

        # Calculate mean of x and y
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        # Calculate the total sum of squares (TSS) and residual sum of squares (RSS)
        ss_total = sum((yi - mean_y) ** 2 for yi in y)
        ss_res = sum((yi - (mean_y + ((xi - mean_x) * (sum((yi - mean_y) * (xi - mean_x) for xi, yi in zip(x, y))) / sum((xi - mean_x) ** 2 for xi in x)))) ** 2 for xi, yi in zip(x, y))

        # Return R^2 value
        return 1 - (ss_res / ss_total)

    # Calculate R^2 for the entire DataFrame
    overall_r2 = calculate_r2(df['Affinity'], df[target_column])
    r2_results = {'Overall': overall_r2}
    
    # Define subset conditions
    subsets = {
        'TQT': df[df['Sequence'].str[4:7] == 'TQT'],
        'SQT': df[df['Sequence'].str[4:7] == 'SQT'],
        'IQT': df[df['Sequence'].str[4:7] == 'IQT'],
        'VQT': df[df['Sequence'].str[4:7] == 'VQT'],
        '!TQT': df[df['Sequence'].str[4:7] != 'TQT'],
        'Full Sequence < 70': df[df['Full Sequence'].str.len() < 70],
        'Full Sequence < 20': df[df['Full Sequence'].str.len() < 20],
        'Full Sequence < 18': df[df['Full Sequence'].str.len() < 18],
        'Full Sequence == 18': df[df['Full Sequence'].str.len() == 18],
        'Full Sequence == 16': df[df['Full Sequence'].str.len() == 16],
        'Full Sequence == 14': df[df['Full Sequence'].str.len() == 14],
        'Full Sequence == 12': df[df['Full Sequence'].str.len() == 12],
        'TQT == 18': df[(df['Full Sequence'].str.len() == 18) & (df['Sequence'].str[4:7] == 'TQT')],
        'TQT == 16': df[(df['Full Sequence'].str.len() == 16) & (df['Sequence'].str[4:7] == 'TQT')],
        'TQT == 14': df[(df['Full Sequence'].str.len() == 14) & (df['Sequence'].str[4:7] == 'TQT')],
        'TQT == 12': df[(df['Full Sequence'].str.len() == 12) & (df['Sequence'].str[4:7] == 'TQT')],
        '!TQT == 16': df[(df['Full Sequence'].str.len() == 16) & (df['Sequence'].str[4:7] != 'TQT')],
        '!TQT == 14': df[(df['Full Sequence'].str.len() == 14) & (df['Sequence'].str[4:7] != 'TQT')],
    }
    
    # Calculate R^2 for each subset
    for key, subset in subsets.items():
        r2_results[key] = calculate_r2(subset['Affinity'], subset[target_column])
    
    return r2_results    

def main():
    affinities_file = 'affinities.csv'
    combined_file = 'combined_analysis_data.csv'
    linked_file = 'linked_analysis_data.csv'

    # Load the data
    affinities_df, combined_df, linked_df = load_data(affinities_file, combined_file, linked_file)

    # Match sequences and calculate relevant statistics
    df_2client, df_1client, df_success_rate = match_sequences(affinities_df, combined_df, linked_df)

    # Save the results to CSV files
    save_dataframes(df_2client, df_1client, df_success_rate)

    # Plot the results
    #plot_results(df_2client, df_1client, df_success_rate)

    # Calculate the R^2 value of results
    success_rate_r2_values = calculate_r2_values(df_success_rate,'Success Rate')
    print(success_rate_r2_values)

    #r2_values = calculate_r2_values(df_1client,'Best Conf. score')
    #print(r2_values)

if __name__ == "__main__":
    main()