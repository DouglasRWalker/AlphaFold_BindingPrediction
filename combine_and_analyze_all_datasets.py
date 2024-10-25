# combine_and_analyze_all_datasets.py

import pandas as pd
from pathlib import Path

def get_binding_status(base_path):
    """Determines the binding status based on the presence of binder.txt or nonbinder.txt files."""
    binding_files = ["binder.txt", "nonbinder.txt", "1equ.txt", "2equ.txt", "3equ.txt", "4equ.txt", "5equ.txt", "6equ.txt", "7equ.txt"]
    for binding_file in binding_files:
        if (base_path / binding_file).exists():
            return binding_file.split('.')[0]
    return "unknown"

def extract_sequence_and_path(base_path, prefix):
    """Extracts the sequence and base path from the corresponding .fasta file."""
    fasta_file = list(base_path.parent.glob(f"{prefix}.fasta"))[0]  # There should be only one match
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
        full_sequence = lines[1].strip()[:-1]  # Second line contains the sequence, remove final character (colon)
    return str(base_path)[47:], full_sequence

def extract_analysis_data(file_path):
    """Extracts the relevant analysis data from an analysis_summary.csv file."""
    #print(file_path)
    df = pd.read_csv(file_path)
    return df

def merge_rows(df):
    """Merge rows based on LC8 dimer and Chain column data."""
    key_columns = ['File Path', 'ClientType', 'WTorAAA', 'BindingStatus', 'Full Sequence', 'Rank', 'Conf. Score']

    merged_data = []
    grouped = df.groupby(key_columns)

    for _, group in grouped:
        skip = []
        for i, row1 in group.iterrows():
            for j, row2 in group.iterrows():
                if len(set(str(row1['LC8 dimer'])) & set(str(row2['LC8 dimer']))) == 2:
                    if row1['Dimer PAE'] > row2['Dimer PAE']:
                        skip.append(i)
                    elif row1['Dimer PAE'] < row2['Dimer PAE']:
                        skip.append(j)

        for i, row1 in group.iterrows():
            if i not in skip:
                if pd.notna(row1['LC8 dimer']):
                    for j, row2 in group.iterrows():
                        if pd.notna(row2['Chain']):
                            if str(row2['Chain']) in str(row1['LC8 dimer']):
                                merged_row = row1.copy()
                                for col in ['Chain', 'Best Binding Region', 'Sequence', 'Average pLDDT', 'Average PAE (L->P)', 'Average PAE (P->L)']:
                                    if col not in key_columns and pd.notna(row2[col]):
                                        merged_row[col] = row2[col]
                                merged_data.append(merged_row)

    return pd.DataFrame(merged_data)

def organize_analysis_data(data_dir, existing_df=None):
    """Organizes analysis data from multiple analysis_summary.csv files."""
    all_data = [] if existing_df is None else [existing_df]

    off_target_file = data_dir / 'off-target_analysis.csv'
    off_target_df = pd.read_csv(off_target_file) if off_target_file.exists() else None

    for summary_file in Path(data_dir).rglob("*analysis_summary.csv"):
        base_path = summary_file.parent
        prefix = base_path.stem.replace("_output", "").replace("results", "")
        file_path, full_sequence = extract_sequence_and_path(base_path, prefix)
        binding_status = get_binding_status(base_path)

        df = extract_analysis_data(summary_file)

        if len(df.columns) == 4:
            print(f'{base_path} contains no interaction data.')
            continue

        # Assign new columns based on the file path
        if '1client' in file_path:
            client_type = '1client'
        elif '2client' in file_path:
            client_type = '2client'
        else:
            client_type = 'Unknown'

        if 'AAA' in file_path:
            wt_or_aaa = 'AAA'
        else:
            wt_or_aaa = 'WT'

        # Insert the new columns
        df.insert(0, 'File Path', file_path)
        df.insert(1, 'ClientType', client_type)
        df.insert(2, 'WTorAAA', wt_or_aaa)
        df.insert(3, 'BindingStatus', binding_status)
        df.insert(4, 'Full Sequence', full_sequence)

        # Merge rows based on LC8 dimer and Chain columns
        df = merge_rows(df)

        # Sort df by descending pLDDT within each distinct set of experiment designators
        df = df.sort_values(by=['Average pLDDT'], ascending=[False])

        grouped = df.groupby(['File Path', 'ClientType', 'WTorAAA', 'BindingStatus', 'Full Sequence'])

        sorted_df = pd.DataFrame()

        for _, group in grouped:
            ranks = group['Rank'].unique()
            for rank in ranks:
                rank_group = group[group['Rank'] == rank]
                sorted_df = pd.concat([sorted_df, rank_group])

        # Check for off-target binding if the file exists
        if off_target_df is not None:
            sorted_df['CheckedStatus'] = ''
            for index, row in sorted_df.iterrows():
                if row['BindingStatus'] in ["binder"]:
                    matching_off_target = off_target_df[
                        (off_target_df['File Path'] == row['File Path']) &
                        (off_target_df['Rank'] == row['Rank']) &
                        (off_target_df['Chain'] == row['Chain']) &
                        (off_target_df['Sequence'] == row['Sequence'])
                    ]
                    #print(f"Matching off-target entries: {matching_off_target}")
                    if not matching_off_target.empty:
                        if matching_off_target.iloc[0]['Off-target'] == 'Off-target':
                            sorted_df.at[index, 'BindingStatus'] = 'Off-target'
                            sorted_df.at[index, 'CheckedStatus'] = ''
                            #print(f"Updated to Off-target for index, {index}, of file {file_path}")
                        else:
                            sorted_df.at[index, 'CheckedStatus'] = ''
                            #print(f"No matching off-target entry or not Off-target for index: {index}")

                    else:
                        sorted_df.at[index, 'CheckedStatus'] = 'NotChecked'

        # Check for duplicates
        if existing_df is not None:
            if not ((existing_df['File Path'] == file_path) & (existing_df['Full Sequence'] == full_sequence)).any():
                all_data.append(sorted_df)
        else:
            all_data.append(sorted_df)

    # Concatenate all the dataframes
    combined_df = pd.concat(all_data, ignore_index=True)

    return combined_df

def create_binding_status_file(df, output_file):
    """Creates a binding status CSV file from the combined DataFrame."""
    binding_status_df = df[['File Path', 'WTorAAA', 'ClientType', 'Full Sequence', 'BindingStatus']].drop_duplicates()
    binding_status_df.to_csv(output_file, index=False)

def sanitize_filename(filename):
    """Sanitizes the filename by removing or replacing invalid characters."""
    return filename.replace(' ', '_').replace('(', '').replace(')', '').replace('>', '').replace('<', '')

def main():
    # Define the directory containing your data files
    data_dir = Path.cwd()  # Set to current working directory
    output_file = "combined_analysis_data.csv"
    binding_status_output_file = "binding_status.csv"

    # Check if the combined CSV file already exists
    if Path(output_file).exists():
        existing_df = pd.read_csv(output_file)
    else:
        existing_df = None

    # Organize the analysis data
    combined_df = organize_analysis_data(data_dir, existing_df)

    # Save the combined dataframe to a CSV file for further analysis and visualization
    combined_df.to_csv(output_file, index=False)
    print(f"Data extraction and organization complete. Data saved to {output_file}")

    # Create the binding status CSV file
    create_binding_status_file(combined_df, binding_status_output_file)
    print(f"Binding status file created and saved to {binding_status_output_file}")

if __name__ == "__main__":
    main()
