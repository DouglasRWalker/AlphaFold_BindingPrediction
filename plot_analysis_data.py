import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sanitize_filename(filename):
    """Sanitizes the filename by removing or replacing invalid characters."""
    return filename.replace(' ', '_').replace('(', '').replace(')', '').replace('>', '').replace('<', '')

def filter_top_20_percent(df):
    """Filters the DataFrame to include only the top 20% of confidence scores."""
    df_top_20_percent = df.groupby(['File Path', 'ClientType', 'WTorAAA']).apply(lambda x: x.nlargest(int(0.2 * len(x)), 'Average pLDDT')).reset_index(drop=True)
    return df_top_20_percent

def filter_top_scores(df):
    """Filters the DataFrame to include only the very best value for each experiment."""
    idx = df.groupby(['File Path', 'ClientType', 'WTorAAA'])['Average pLDDT'].idxmax()
    df_top_value = df.loc[idx].reset_index(drop=True)
    return df_top_value

def filter_by_1client_pLDDT(df):
    """Filters the DataFrame to include only rows with value for 1client Average pLDDT above 75."""
    df_above_75 = df[df['1client_Average pLDDT'] > 90]
    return df_above_75

def plot_histograms_and_pairplot(df, client_type, parameters, ranges, suffix=''):
    """Plots histograms for each scoring parameter and a pair plot."""
    sns.set(style="whitegrid")

    # Filter the dataframe by client type and binding status
    df_filtered = df[(df['ClientType'] == client_type)]

    # Increase plot size
    plot_size = (15, 9)  # 150% of the default size

    # Create a histogram for each scoring parameter
    for param, range_values in zip(parameters, ranges):
        plt.figure(figsize=plot_size)
        data = df_filtered[[param, 'BindingStatus']]
        binders = data[data['BindingStatus'] == 'binder'][param]
        non_binders = data[data['BindingStatus'] == 'nonbinder'][param]
        offtarget = data[data['BindingStatus'] == 'Off-target'][param]

        bins = np.linspace(range_values[0], range_values[1], 51)  # 50 bins

        # Calculate the histogram data
        binders_hist, _ = np.histogram(binders, bins=bins)
        non_binders_hist, _ = np.histogram(non_binders, bins=bins)
        unknowns_hist, bin_edges = np.histogram(offtarget, bins=bins)

        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Plot histograms side by side
        width = (bin_edges[1] - bin_edges[0]) / 4

        plt.bar(bin_centers - width, binders_hist, width=width, color='cyan', alpha=0.7, label='Binder', align='center')
        plt.bar(bin_centers, non_binders_hist, width=width, color='orange', alpha=0.7, label='Non-binder', align='center')
        plt.bar(bin_centers + width, unknowns_hist, width=width, color='magenta', alpha=0.7, label='Off-target', align='center')

        plt.title(f'{param} Histogram for {client_type} {suffix}')
        plt.xlabel(param)
        plt.ylabel('Count')
        plt.legend(title='Binding Status')
        sanitized_param = sanitize_filename(param)
        plt.savefig(f'{sanitized_param}_{client_type}_histogram{suffix}.png')
        plt.close()

    # Create a pair plot for the scoring parameters (Binders and Non-binders)
    df_pairplot = df_filtered[df_filtered['BindingStatus'].isin(['binder', 'nonbinder', 'Off-target'])]
    
    g = sns.PairGrid(df_pairplot, vars=parameters, hue='BindingStatus', 
                     palette={'binder': 'cyan', 'nonbinder': 'orange', 'Off-target': 'magenta'})
    g = g.map_upper(sns.kdeplot, thresh=0.2, levels=5)
    g = g.map_diag(sns.histplot)
    g = g.map_lower(sns.kdeplot, thresh=0.2, levels=5)
    
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)
    
    for row in range(len(g.axes)):
        for col in range(len(g.axes[row])):
            if col != 0:
                g.axes[row, col].set_ylabel('')
            if row != len(g.axes) - 1:
                g.axes[row, col].set_xlabel('')

    plt.suptitle(f'Pair Plot for {client_type} {suffix}', y=1.02)
    plt.savefig(f'{client_type}_pair_plot{suffix}.png')
    plt.close()

'''    # Create a pair plot for the unknown binding statuses
    df_pairplot_unknown = df_filtered[df_filtered['BindingStatus'] == 'Unknown']
    if not df_pairplot_unknown.empty:
        g_unknown = sns.PairGrid(df_pairplot_unknown, vars=parameters, hue='BindingStatus', 
                                 palette={'Unknown': 'magenta'})
        g_unknown = g_unknown.map_upper(sns.kdeplot, thresh=0.2, levels=5)
        g_unknown = g_unknown.map_diag(sns.histplot)
        g_unknown = g_unknown.map_lower(sns.kdeplot, thresh=0.2, levels=5)
        
        for i, j in zip(*np.triu_indices_from(g_unknown.axes, 1)):
            g_unknown.axes[i, j].set_visible(False)
        
        for row in range(len(g_unknown.axes)):
            for col in range(len(g_unknown.axes[row])):
                if col != 0:
                    g_unknown.axes[row, col].set_ylabel('')
                if row != len(g_unknown.axes) - 1:
                    g_unknown.axes[row, col].set_xlabel('')

        plt.suptitle(f'Pair Plot for Unknown Binding Status for {client_type} {suffix}', y=1.02)
        plt.savefig(f'{client_type}_pair_plot_unknown{suffix}.png')
        plt.close()
'''

def calculate_false_positives_negatives(df, scoring_parameters, cutoff):
    """Calculate false positives and false negatives based on scoring parameters and ranges."""
    # Define which parameters have upper or lower limit cut-offs
    upper_limit_parameters = ['Average PAE (L->P)', 'Average PAE (P->L)', 'Dimer PAE', '1client_Average PAE (L->P)', '1client_Average PAE (P->L)', '1client_Dimer PAE']
    
    cutoffs = {param: cutoff for param, cutoff in zip(scoring_parameters, cutoff)}
    
    # Initialize lists to hold false positives and negatives
    false_positives = []
    false_negatives = []

    positive_nonbinders = 0
    for index, row in df.iterrows():
        is_false_positive = False
        is_false_negative = False

        if row['BindingStatus'] == 'nonbinder':
            is_false_positive = True
        if row['BindingStatus'] == 'Off-target':
            is_false_positive = True

        for param, cutoff in cutoffs.items():
            if param in upper_limit_parameters:
                # Upper limit cut-off
                if row['BindingStatus'] == 'nonbinder' and row[param] > cutoff:
                    is_false_positive = False
                elif row['BindingStatus'] == 'Off-target' and row[param] > cutoff:
                    is_false_positive = False
                elif row['BindingStatus'] == 'binder' and row[param] > cutoff:
                    is_false_negative = True
            else:
                # Lower limit cut-off
                if row['BindingStatus'] == 'nonbinder' and row[param] < cutoff:
                    is_false_positive = False
                elif row['BindingStatus'] == 'Off-target' and row[param] < cutoff:
                    is_false_positive = False
                elif row['BindingStatus'] == 'binder' and row[param] < cutoff:
                    is_false_negative = True

        if is_false_positive:
            false_positives.append(row)
            if row['BindingStatus'] == 'nonbinder':
                positive_nonbinders+=1
        if is_false_negative:
            false_negatives.append(row)

    false_positives_df = pd.DataFrame(false_positives)
    false_negatives_df = pd.DataFrame(false_negatives)

    # Save the false positives and negatives to CSV files
    false_positives_df.to_csv('false_positives.csv', index=False)
    false_negatives_df.to_csv('false_negatives.csv', index=False)

    # Calculate false positive and false negative rates
    total_nonbinders = len(df[df['BindingStatus'] == 'nonbinder'])
    total_binders = len(df[df['BindingStatus'] == 'binder'])
    total_Offtarget = len(df[df['BindingStatus'] == 'Off-target'])
    
    false_pos_nonbinder_rate = positive_nonbinders / total_nonbinders
    false_positive_rate = len(false_positives_df) / (total_nonbinders + total_Offtarget)
    false_negative_rate = len(false_negatives_df) / total_binders

    print(f"False pos(w/out off): {false_pos_nonbinder_rate:.4f}")
    print(f"False positive rate:  {false_positive_rate:.4f}")
    print(f"False negative rate:  {false_negative_rate:.4f}")

    return false_positives_df, false_negatives_df

def main():
    #input_file = "combined_analysis_data.csv"
    #combined_df = pd.read_csv(input_file)


    # Define the scoring parameters and their ranges
    scoring_parameters = ['Conf. Score', 'Average pLDDT', 'Average PAE (L->P)', 'Average PAE (P->L)', 'Dimer PAE']
    ranges = [(0, 1), (0, 100), (0, 37.5), (0, 37.5), (0,37.5)]

    # Plot the histograms and pair plot for the combined data
    #for client_type in ['1client', '2client']:
    #    plot_histograms_and_pairplot(combined_df, client_type, scoring_parameters, ranges, suffix='')

    # Filter the data to include only the top 20% of confidence scores
    #combined_df_top_20 = filter_top_20_percent(combined_df)

    # Plot the histograms and pair plot for the top 20% of confidence scores
    #plot_histograms_and_pairplot(combined_df_top_20, '1client', scoring_parameters, ranges, suffix='_top_20')
    #plot_histograms_and_pairplot(combined_df_top_20, '2client', scoring_parameters, ranges, suffix='_top_20')

    # Filter the data to include only the top 1 result of predictions
    #combined_df_top_1 = filter_top_scores(combined_df)

    # Plot the histograms and pair plot for the top 20% of confidence scores
    #plot_histograms_and_pairplot(combined_df_top_1, '1client', scoring_parameters, ranges, suffix='_top_1')
    #plot_histograms_and_pairplot(combined_df_top_1, '2client', scoring_parameters, ranges, suffix='_top_1')


    input_file = "linked_analysis_data.csv"
    combined_df = pd.read_csv(input_file)

    #Define scoring parameters and ranges for linked pair plot
    scoring_parameters = ['Conf. Score', 'Average pLDDT', 'Average PAE (L->P)', 'Average PAE (P->L)', 'Dimer PAE', '1client_Conf. Score', '1client_Average pLDDT', '1client_Average PAE (L->P)', '1client_Average PAE (P->L)', '1client_Dimer PAE']
    ranges = [(0, 1), (0, 100), (0, 37.5), (0, 37.5), (0,37.5), (0, 1), (0, 100), (0, 37.5), (0, 37.5), (0,37.5)]

    #plot_histograms_and_pairplot(combined_df, '2client', scoring_parameters, ranges, suffix='_Linked_plots')

    # Filter the data to include only the top 20% of confidence scores
    #combined_df_top_20 = filter_top_20_percent(combined_df)

    # Plot the histograms and pair plot for the top 20% of confidence scores
    #plot_histograms_and_pairplot(combined_df_top_20, '2client', scoring_parameters, ranges, suffix='_Linked_plots_top_20')

    # Filter the data to include only the top Average pLDDT scores
    combined_df_top_1 = filter_top_scores(combined_df)

    # Plot the histograms and pair plot for the top 20% of confidence scores
    #plot_histograms_and_pairplot(combined_df_top_1, '2client', scoring_parameters, ranges, suffix='_Linked_plots_top_1')

    # Filter the data to include only the top Average pLDDT scores
    #combined_df_pLDDT_filter = filter_by_1client_pLDDT(combined_df_top_1)

    # Plot the histograms and pair plot for the top 20% of confidence scores
    #plot_histograms_and_pairplot(combined_df_pLDDT_filter, '2client', scoring_parameters, ranges, suffix='_Linked_top_1_1client_pLDDT')

    cutoff = [0.7, 79.0, 2.68, 5.81, 1.24, 0.8, 62.0, 6.3, 9.4, 1.1]

    false_positives_df, false_negatives_df = calculate_false_positives_negatives(combined_df_top_1, scoring_parameters, cutoff)


if __name__ == "__main__":
    main()