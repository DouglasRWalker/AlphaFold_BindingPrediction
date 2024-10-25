import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import glob

# Define threshold values based on client type
thresholds = {
    'Conf. Score': 0.7,
    'Average pLDDT': 79.0,
    'Average PAE (L->P)': 2.68,
    'Average PAE (P->L)': 5.81,
    'Dimer PAE': 1.24,
    '1client_Conf. Score': 0.8,
    '1client_Average pLDDT': 62.0,
    '1client_Average PAE (L->P)': 6.3,
    '1client_Average PAE (P->L)': 9.4,
    '1client_Dimer PAE': 1.1
}

# Step 1: Load linked_analysis_data.csv
df = pd.read_csv('linked_analysis_data.csv')

# Step 2: Filter entries with "unknown" in the BindingStatus
unknown_binders = df[df['BindingStatus'] == 'unknown']

# Step 3: Group by File Path and select the structure with the highest pLDDT_avg
best_structures = unknown_binders.loc[unknown_binders.groupby('File Path')['Average pLDDT'].idxmax()]

# Step 4: Extract Protein Name and Protein Length from File Path
def extract_protein_info(filepath):
    split_path = filepath.split("\\")
    
    # Ensure that the split by "_" has at least two parts
    try:
        protein_name = split_path[-2]  # Protein name
        filename_parts = split_path[-1].split("_")
        
        # Ensure the filename has the expected format with "_"
        if len(filename_parts) < 2:
            raise ValueError(f"Unexpected file name format in {filepath}")

        # Extract range of residues from the filename after "_"
        start_end = filename_parts[1].split("-")
        
        # Ensure start and end exist and are numeric
        start = int(start_end[0])
        end = int(start_end[1])
        
        return protein_name, start, end
    
    except (IndexError, ValueError) as e:
        # Catch index errors or value errors due to invalid format
        #print(f"Error processing {filepath}: {e}")
        return None  # Return None to indicate invalid entry

# Apply the function to extract protein info and filter out invalid rows
best_structures[['Protein', 'Start', 'End']] = best_structures['File Path'].apply(
    lambda x: pd.Series(extract_protein_info(x)) if extract_protein_info(x) else pd.Series([None, None, None])
)

# Remove rows with None values (invalid entries)
best_structures = best_structures.dropna(subset=['Protein', 'Start', 'End'])


# Calculate total protein length by taking the max of "End" values for each protein
protein_lengths = best_structures.groupby('Protein')['End'].max().reset_index()
protein_lengths.columns = ['Protein', 'Total Length']

# Step 5: Check binding prediction using thresholds
def predict_binder(row):
    if row['ClientType'] == '2client':
        return (
            row['Average pLDDT'] >= thresholds['Average pLDDT'] and
            row['Average PAE (L->P)'] <= thresholds['Average PAE (L->P)'] and
            row['Average PAE (P->L)'] <= thresholds['Average PAE (P->L)'] and
            row['1client_Average pLDDT'] >= thresholds['1client_Average pLDDT'] and
            row['1client_Average PAE (L->P)'] <= thresholds['1client_Average PAE (L->P)'] and
            row['1client_Average PAE (P->L)'] <= thresholds['1client_Average PAE (P->L)'] and
            row['1client_Dimer PAE'] <= thresholds['1client_Dimer PAE']
        )
    return False

best_structures['PredictedBinder'] = best_structures.apply(predict_binder, axis=1)

# Step 6: Report the length of the full sequence and the contents of the "Sequence" column
best_structures['Full Sequence Length'] = best_structures['Full Sequence'].apply(len)
best_structures['Anchor Motif'] = best_structures['Sequence']#.apply(lambda x: x[5:8])

# Step 7: Extrapolate affinity based on anchor motif and peptide length (placeholder)
def extrapolate_affinity(row):
    if row['PredictedBinder']:
        length = row['Full Sequence Length']
        anchor_motif = row['Anchor Motif']
        # Example affinity prediction based on these two factors (placeholder logic)
        if anchor_motif == 'TQT':
            return 100 / length  # Example: affinity based on length and motif
    return None

best_structures['PredictedAffinity'] = best_structures.apply(extrapolate_affinity, axis=1)

# Step 8: Visual representation of binding sites
def plot_binding_sites(protein_length, binding_sites, protein_name, folder):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot([0, protein_length], [1, 1], color='grey', lw=6, label='Protein', solid_capstyle='butt')

    for site in binding_sites:
        start, end = site['start'], site['end']
        ax.plot([start, end], [1, 1], color='orange', lw=10, label='Binding Site', solid_capstyle='butt')

 # Remove y-axis ticks and labels
    ax.set_yticks([])

    # Remove outer box (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Set the x-axis limits to the protein length
    ax.set_xlim(0, protein_length)
    ax.set_ylim(0.999,1.005)
    ax.set_xlabel('Protein Position')
    ax.set_title('Predicted Binding Sites on ' + protein_name, pad=3)
    plt.savefig(folder + '/Predicted ' + protein_name+ ' Binding sites.png')
    plt.close()

# Example of visualizing for a specific protein
def visualize_protein_binding(protein_name, best_structures, folder):
    protein_structures = best_structures[best_structures['Protein'] == protein_name].copy()

    protein_length = protein_lengths[protein_lengths['Protein'] == protein_name]['Total Length'].values[0]

    # Gather binding site start and end positions
    binding_sites = []
    for _, row in protein_structures.iterrows():
        local_start = int(row['Best Binding Region'].split(",")[1].split("-")[0])
        local_end = int(row['Best Binding Region'].split(",")[1].split("-")[1])
        binding_sites.append({'start': row['Start']+local_start, 'end': row['Start']+local_end})

    plot_binding_sites(protein_length, binding_sites, protein_name, folder)

    LPred = pd.read_csv('LC8Pred_results.csv')

    LP_binding_sites = []
    anchors = []
    pot_anchors = []
    if not LPred[LPred['Protein'] == protein_name]['Sure LC8Pred Anchors'].empty:
        if not isinstance(LPred[LPred['Protein'] == protein_name]['Sure LC8Pred Anchors'].iloc[0],float):
            anchors = LPred[LPred['Protein'] == protein_name]['Sure LC8Pred Anchors'].iloc[0].split(";")
        else:
            anchors.append(LPred[LPred['Protein'] == protein_name]['Sure LC8Pred Anchors'])

    if not LPred[LPred['Protein'] == protein_name]['Pred LC8Pred Anchors'].empty:
        if not isinstance(LPred[LPred['Protein'] == protein_name]['Pred LC8Pred Anchors'].iloc[0],float):
            pot_anchors = LPred[LPred['Protein'] == protein_name]['Pred LC8Pred Anchors'].iloc[0].split(";")
        else:
            pot_anchors.append(LPred[LPred['Protein'] == protein_name]['Pred LC8Pred Anchors'])

    for a in anchors:
        if not math.isnan(float(a)):
            start = int(a)-5
            end = int(a)+2
            LP_binding_sites.append({'start': start, 'end': end})

    for p in pot_anchors:
        if not math.isnan(float(p)):
            start = int(p)-5
            end = int(p)+2
            LP_binding_sites.append({'start':start, 'end': end})

    plot_binding_sites(protein_length, LP_binding_sites, protein_name, 'LC8_binding_sites')
    

# Save the predictions to a new CSV where PredictedBinder is True
predicted_binders = best_structures[best_structures['PredictedBinder']]

# Define the function to parse PDB and get alpha carbon coordinates
def parse_pdb(pdb_file):
    """Parses a PDB file and returns a dictionary of alpha carbon coordinates keyed by chain-residue."""
    ca_coords = {}
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain = line[21].strip()
                resi = line[22:26].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ca_coords[(chain, int(resi))] = (x, y, z)
    return ca_coords

# Function to calculate Euclidean distance between two points
def get_distance(coord1, coord2):
    """Calculates the Euclidean distance between two 3D coordinates."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)

# Function to find the anchor residue based on the distance to Chain C, residue F62
def find_anchor_residue(ca_coords):
    """Find the peptide residue closest to Chain C, residue F62, and return its surrounding residues."""
    target_residue = ('C', 62)  # Chain C, residue 62
    if target_residue not in ca_coords:
        print("Target residue not found in structure.")
        return None, None, None

    target_coord = ca_coords[target_residue]

    # Find the peptide chain residues closest to the target residue
    min_dist = float('inf')
    anchor_residue = None

    for (chain, resi), coord in ca_coords.items():
        if chain in ('A', 'B'):  # Assuming 'A' or 'B' is the peptide chain
            dist = get_distance(target_coord, coord)
            if dist < min_dist:
                min_dist = dist
                anchor_residue = resi-1

    if anchor_residue is None:
        print("No peptide chain found near the target residue.")
        return None, None, None

    # Return anchor residue (-1, 0, +1)
    return anchor_residue

# Step 9: Iterate through predicted_binders to extract anchor sequence
def extract_anchor_sequences(row):
    # Construct the PDB file path using File Path and Rank
    pdb_pattern = os.path.join(row['File Path'], f"*{row['Rank']}*.pdb")
    pdb_files = glob.glob(pdb_pattern)  # Match files using the pattern

    if len(pdb_files) == 0:
        print(f"No PDB file found for pattern: {pdb_pattern}")
        return None

    pdb_file = pdb_files[0]  # Assuming the first match is the correct one
    ca_coords = parse_pdb(pdb_file)  # Get alpha carbon coordinates
    res_0 = find_anchor_residue(ca_coords)

    if res_0 is None:
        return None  # If no anchor found, return None

    # Use the row parameter to extract the substring from 'Full Sequence'
    anchor_seq = row['Full Sequence'][res_0-2:res_0+1]  # Correctly access the Full Sequence for the current row
    return anchor_seq  # Make sure to return the anchor sequence

predicted_binders.loc[:, 'Anchor Motif'] = predicted_binders.apply(extract_anchor_sequences, axis=1)

def extrapolate_affinity(row):
    if row['PredictedBinder']:
        length = row['Full Sequence Length']
        anchor_motif = row['Anchor Motif']
        # Example affinity prediction based on these two factors (placeholder logic)
        if anchor_motif == 'TQT':
            return (row['1client_Dimer PAE']-0.89)/0.0035  # Example: affinity based on length and motif
        else:
            #return (row['1client_Average PAE (P->L)']-1.9404)/0.045
            return (row['Avg_1client_Average PAE (P->L)']-4.4494)/0.1351
    return None


df_grouped = df.groupby('File Path')['1client_Average PAE (P->L)'].mean().reset_index()
predicted_binders.loc[:, 'Avg_1client_Average PAE (P->L)'] = 0.0

df_grouped = df_grouped.set_index('File Path')
predicted_binders = predicted_binders.set_index('File Path')
predicted_binders['Avg_1client_Average PAE (P->L)'] = df_grouped['1client_Average PAE (P->L)']
predicted_binders = predicted_binders.reset_index()


predicted_binders.loc[:,'PredictedAffinity'] = predicted_binders.apply(extrapolate_affinity, axis=1)


predicted_binders.to_csv('predicted_binders_with_affinity_inclusive.csv', index=False)

for _, protein in protein_lengths.iterrows():
    visualize_protein_binding(protein['Protein'], predicted_binders, "Pred_binding_figures_inclusive")

predicted_binders.loc[predicted_binders['PredictedAffinity'] < 1, 'PredictedAffinity'] =1
predicted_binders = predicted_binders[predicted_binders['PredictedAffinity'] <=40]

predicted_binders.to_csv('predicted_binders_with_affinity_inclusive_filtered.csv', index=False)

for _, protein in protein_lengths.iterrows():
    visualize_protein_binding(protein['Protein'], predicted_binders, "Pred_binding_figures_inclusive_filtered")

#END