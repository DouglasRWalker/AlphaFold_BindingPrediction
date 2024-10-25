import pandas as pd
from pathlib import Path

def link_clients(data_file, output_file):
    # Read the combined analysis data
    combined_df = pd.read_csv(data_file)

    # Group by 'Full Sequence' and then subgroup by 'ClientType'
    grouped = combined_df.groupby(['Full Sequence'])

    linked_data = []

    # Iterate through each group
    for sequence, group in grouped:
        group_2client = group[group['ClientType'] == '2client'].sort_values(by=['Average pLDDT'], ascending=[False])
        ranks = group_2client[['File Path', 'Rank']]['Rank'].unique()
        sorted_2client = pd.DataFrame()
        for rank in ranks:
            rank_group = group_2client[group_2client['Rank'] == rank]
            sorted_2client = pd.concat([sorted_2client, rank_group])
        group_2client = sorted_2client
        group_1client = group[group['ClientType'] == '1client'].sort_values(by=['Average pLDDT'], ascending=[False])
        ranks = group_1client[['File Path', 'Rank']]['Rank'].unique()
        sorted_1client = pd.DataFrame()
        for rank in ranks:
            rank_group = group_1client[group_1client['Rank'] == rank]
            sorted_1client = pd.concat([sorted_1client, rank_group])
        group_1client = sorted_1client
        
        if group_2client.empty or group_1client.empty:
            continue
        
        used_1client_rows = set()
        
        i = 0
        while i < len(group_2client):
            row_2client = group_2client.iloc[i]
            matched = False
            for j, row_1client in group_1client.iterrows():
                if j in used_1client_rows:
                    continue
                if row_2client['BindingStatus'] == row_1client['BindingStatus']:
                    # Link the 1client entry to the 2client entry
                    linked_row = row_2client.tolist() + row_1client.tolist()
                    linked_data.append(linked_row)
                    used_1client_rows.add(j)
                    matched = True
                    # Check if the next 2client row has the same rank
                    if i + 1 < len(group_2client) and group_2client.iloc[i + 1]['Rank'] == row_2client['Rank']:
                        i += 1
                        row_2client_next = group_2client.iloc[i]
                        linked_row = row_2client_next.tolist() + row_1client.tolist()
                        linked_data.append(linked_row)
                    break
            if not matched:
                # If no match found, append the 2client entry with NaNs for 1client fields
                linked_row = row_2client.tolist() + [pd.NA] * len(row_1client)
                linked_data.append(linked_row)
            i += 1

    # Create a new DataFrame for linked data
    columns = list(group_2client.columns) + [f'1client_{col}' for col in group_1client.columns]
    linked_df = pd.DataFrame(linked_data, columns=columns)

    # Save the linked data to a CSV file
    linked_df.to_csv(output_file, index=False)
    print(f"Linked data saved to {output_file}")

if __name__ == "__main__":
    data_file = "combined_analysis_data.csv"
    output_file = "linked_analysis_data.csv"
    link_clients(data_file, output_file)