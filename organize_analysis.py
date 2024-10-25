import os
import re
import sys
import pandas as pd
from collections import defaultdict

def parse_output_file(filepath, structured_chains):
    """Parses an output file to extract relevant information."""
    with open(filepath, 'r') as file:
        content = file.readlines()

    # Extract the confidence score
    confidence_score = float(content[0].split(":")[1].strip())

    # Initialize variables to store binding regions and PAE values
    binding_regions = []
    structured_interactions = defaultdict(list)

    i = 0
    in_close_residues_section = False
    while i < len(content):
        line = content[i].strip()

        if line.startswith("Pairs of close residues"):
            in_close_residues_section = True
            i += 1
            continue

        if in_close_residues_section:
            if not line:
                in_close_residues_section = False
            else:
                parts = line.split()
                chain1 = parts[0][0]
                chain2 = parts[0].split('-')[1][0]
                residue1 = parts[0].split(',')[1].split('-')[0]
                residue2 = parts[0].split('-')[1].split(',')[1].split(':')[0]
                pae1 = float(parts[1])
                pae2 = float(parts[3])
                avg_pae = (pae1 + pae2) / 2.0

                if chain1 in structured_chains and chain2 in structured_chains and residue1 != "G89" and residue2 != "G89":
                    structured_interactions[(chain1, chain2)].append(avg_pae)
            i += 1
            continue

        if line.startswith("Binding region in LC8 chain"):
            chain_info = line.split()[-1][:-1]
            structured_chain = chain_info
            
            # Move to the next line to read binding region details
            i += 1
            region_info = content[i].strip()
            parts = region_info.split(": ")
            binding_seq = parts[1].split(" - ")[0]
            disordered_chain = parts[0].split(",")[0]
            numeric_sequence = parts[0].split(",")[1]
            avg_pae_lp = float(parts[-1])
            avg_pae_pl = float(parts[-2].split(" ")[0])
            
            # Move to the next line to read PLDDT score
            i += 1
            avg_plddt = float(content[i].strip().split(":")[1].strip())

            # Store the binding region information
            binding_regions.append((structured_chain, binding_seq, avg_pae_lp, avg_pae_pl, avg_plddt, disordered_chain, numeric_sequence))
        
        i += 1

    return confidence_score, binding_regions, structured_interactions

def find_best_binding_regions(binding_regions):
    """Finds the best binding regions based on the lowest L->P Average PAE value."""
    best_regions = {}
    
    for chain, seq, avg_pae_lp, avg_pae_pl, avg_plddt, disordered_chain, numeric_sequence in binding_regions:
        if chain not in best_regions:
            best_regions[chain] = []
        best_regions[chain].append((seq, avg_pae_lp, avg_pae_pl, avg_plddt, disordered_chain, numeric_sequence))
    
    # Select the best binding region for each chain
    for chain in best_regions:
        best_regions[chain].sort(key=lambda x: x[1])  # Sort by avg_pae_lp
        best_regions[chain] = best_regions[chain][0]  # Select the best one
    
    return best_regions

def organize_analysis(output_directory, structured_chains):
    results = []
    
    for output_file in os.listdir(output_directory):
        if output_file.endswith('_interaction_analysis.txt'):
            filepath = os.path.join(output_directory, output_file)
            rank_part = output_file.split('_interaction_analysis.txt')[0]

            confidence_score, binding_regions, structured_interactions = parse_output_file(filepath, structured_chains)
            best_binding_regions = find_best_binding_regions(binding_regions)

            for (chain1, chain2), pae_values in structured_interactions.items():
                avg_pae = sum(pae_values) / len(pae_values)
                results.append({
                    "Rank": rank_part,
                    "Conf. Score": confidence_score,
                    "LC8 dimer": f"{chain1}-{chain2}",
                    "Dimer PAE": avg_pae
                })     
       
            for chain, (seq, avg_pae_lp, avg_pae_pl, avg_plddt, disordered_chain, numeric_sequence) in best_binding_regions.items():
                results.append({
                    "Rank": rank_part,
                    "Conf. Score": confidence_score,
                    "Chain": chain,
                    "Best Binding Region": f"{disordered_chain},{numeric_sequence}",
                    "Sequence": f"{seq}",
                    "Average pLDDT": avg_plddt,
                    "Average PAE (L->P)": avg_pae_lp,
                    "Average PAE (P->L)": avg_pae_pl
                })

    results_df = pd.DataFrame(results)
    summary_filepath = os.path.join(output_directory, f"{output_directory[:-6]}analysis_summary.csv")
    results_df.to_csv(summary_filepath, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python organize_analysis.py output_directory")
        sys.exit(1)
    
    structured_chains = 'CDEFGHIJKLMNOPQRSTUVWXYZ'
    output_directory = sys.argv[1]
    organize_analysis(output_directory, structured_chains)
    print(f"Summary written for {output_directory}")
