import sys
import json
import math
import numpy as np
import os
import re

def read_fasta(filename):
    """Reads a fasta file and returns a list of sequences."""
    with open(filename, 'r') as file:
        content = file.read()
    sequences = content.split('\n')
    sequences = [seq.replace(':', '') for seq in sequences if seq]
    sequences = sequences[1:]
    return sequences


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


def get_distance(coord1, coord2):
    """Calculates the Euclidean distance between two 3D coordinates."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)


def get_close_residues(ca_coords, distance_threshold=5.0):
    """Returns a list of residue pairs that are within a certain distance threshold (5 Å) between alpha carbons, considering only inter-chain distances."""
    close_residues = []
    keys = list(ca_coords.keys())
    for i, key1 in enumerate(keys):
        for key2 in keys[i+1:]:
            chain1, res1 = key1
            chain2, res2 = key2
            if chain1 != chain2:  # Ensure inter-chain comparison
                dist = get_distance(ca_coords[key1], ca_coords[key2])
                if dist < distance_threshold:
                    close_residues.append((key1, key2))
    return close_residues


def analyze_json(json_file):
    """Extracts plddt and paes from the json file."""
    with open(json_file, 'r') as file:
        data = json.load(file)
    plddts = data['plddt']
    paes = np.array(data['pae'])
    ptm = data['ptm']
    iptm = data['iptm']
    conf = 0.8*iptm+0.2*ptm
    return plddts, paes, conf


def create_residue_index_mapping(sequences):
    """Creates a mapping of residue indices to their corresponding chain and position."""
    residue_index_mapping = {}
    chain_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    current_index = 0
    for chain_index, sequence in enumerate(sequences):
        chain = chain_letters[chain_index]
        for resi in range(1, len(sequence) + 1):
            residue_index_mapping[(chain,resi)] = current_index
            current_index += 1
            
    return residue_index_mapping


def create_residue_to_aa_mapping(sequences):
    """Creates a mapping of each chain and residue to the corresponding amino acid."""
    residue_to_aa_mapping = {}
    chain_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for chain_index, sequence in enumerate(sequences):
        chain = chain_letters[chain_index]
        for resi in range(1, len(sequence) + 1):
            residue_to_aa_mapping[(chain, resi)] = sequence[resi - 1]
            
    return residue_to_aa_mapping


def find_binding_regions(ca_coords, close_residues, structured_chains, binding_range=(61, 70), stretch_length=8, distance_threshold=7.5):
    """Identifies binding regions around close residues within structured chains."""
    binding_regions = set()  # Use a set to automatically handle duplicates
    
    for (chain1, resi1), (chain2, resi2) in close_residues:
        if chain1 in structured_chains and binding_range[0] <= resi1 <= binding_range[1] and chain2 not in structured_chains:
            # chain1 is structured, chain2 is disordered
            for offset in range(-stretch_length + 1, 1):
                stretch = [(chain2, resi2 + offset + i) for i in range(stretch_length) if (chain2, resi2 + offset + i) in ca_coords]
                if len(stretch) == stretch_length:
                    # Check if all residues in the stretch are within the distance threshold
                    if all(any(get_distance(ca_coords[res], ca_coords[(chain1, resi1_binding)]) < distance_threshold for resi1_binding in range(binding_range[0], binding_range[1] + 1) if (chain1, resi1_binding) in ca_coords) for res in stretch):
                        binding_regions.add((chain1, tuple(stretch)))
        elif chain2 in structured_chains and binding_range[0] <= resi2 <= binding_range[1] and chain1 not in structured_chains:
            # chain2 is structured, chain1 is disordered
            for offset in range(-stretch_length + 1, 1):
                stretch = [(chain1, resi1 + offset + i) for i in range(stretch_length) if (chain1, resi1 + offset + i) in ca_coords]
                if len(stretch) == stretch_length:
                    # Check if all residues in the stretch are within the distance threshold
                    if all(any(get_distance(ca_coords[res], ca_coords[(chain2, resi2_binding)]) < distance_threshold for resi2_binding in range(binding_range[0], binding_range[1] + 1) if (chain2, resi2_binding) in ca_coords) for res in stretch):
                        binding_regions.add((chain2, tuple(stretch)))
    #print(binding_regions)
    return list(binding_regions)


def write_output(filename, residue_pairs, paes, mapping, binding_regions, residue_to_aa_mapping, plddts, conf):
    """Writes the residue pairs and their corresponding pae values to a file."""
    with open(filename, 'w') as file:
        file.write(f"Confidence Score: {conf:.2f}\n")
        file.write("\nPairs of close residues\n")
        for pair in residue_pairs:
            chain_res1, chain_res2 = pair
            index1 = mapping[chain_res1]
            index2 = mapping[chain_res2]
            pae_value = paes[index1, index2]  # Using numpy array indexing
            pae_value_mirror = paes[index2, index1]
            aa1 = residue_to_aa_mapping[chain_res1]
            aa2 = residue_to_aa_mapping[chain_res2]
            file.write(f"{chain_res1[0]},{aa1}{chain_res1[1]}-{chain_res2[0]},{aa2}{chain_res2[1]}: {pae_value} and {pae_value_mirror}\n")

        file.write("\nBinding Regions:\n")
        for structured_chain, stretch in binding_regions:
            total_pae_binding = 0
            total_pae_62_88 = 0
            total_pae_entire = 0
            total_pae_binding_mirror = 0
            total_pae_62_88_mirror = 0
            total_pae_entire_mirror = 0
            count_binding = 0
            count_62_88 = 0
            count_entire = 0
            count_binding_mirror = 0
            count_62_88_mirror = 0
            count_entire_mirror = 0
            total_plddt_binding = 0

            file.write(f"Binding region in LC8 chain {structured_chain}:\n")
            for chain, res in stretch:
                for i in range(62, 70):
                    if (structured_chain, i) in mapping:
                        pae_value = paes[mapping[(chain, res)], mapping[(structured_chain, i)]]
                        pae_value_mirror = paes[mapping[(structured_chain, i)], mapping[(chain, res)]]
                        total_pae_binding += pae_value
                        total_pae_binding_mirror += pae_value_mirror
                        total_plddt_binding += plddts[mapping[(chain, res)]]
                        count_binding += 1
                        count_binding_mirror += 1
            avg_pae = total_pae_binding / count_binding if count_binding > 0 else float('nan')
            avg_pae_mirror = total_pae_binding_mirror / count_binding_mirror if count_binding_mirror > 0 else float('nan')
            avg_plddt = total_plddt_binding / count_binding if count_binding > 0 else float('nan')
            file.write(f"{chain},{stretch[0][1]}-{stretch[-1][1]}: {''.join([residue_to_aa_mapping[(chain, res)] for chain, res in stretch])} - Average PAE:: P->L: {avg_pae:.3f} and L->P: {avg_pae_mirror:.3f}\n")
            file.write(f"Average PLDDT: {avg_plddt: .3f}\n")

            # Print the full matrix of PAE values for the specified range
            file.write("PAE matrix for the binding region:\n")

            # Print the column labels
            col_labels = [f"{residue_to_aa_mapping[(structured_chain, i)]}{i}" for i in range(62, 70)]
            col_labels_mirror = [f"{residue_to_aa_mapping[(chain, res)]}{res}" for chain, res in stretch]
            file.write(" " * 7 + "  ".join(col_labels) + " " * 12 + "  ".join(col_labels_mirror) + "\n")

            row_values = [[0 for _ in range(62,70)] for _ in range(len(stretch))]
            row_values_mirror = [[0 for _ in range(len(stretch))] for _ in range(62,70)]
            for j, (chain, res) in enumerate(stretch):
                for k,i in enumerate(range(62, 70)):
                    if (structured_chain, i) in mapping:
                        pae_value = paes[mapping[(chain, res)], mapping[(structured_chain, i)]]
                        pae_value_mirror = paes[mapping[(structured_chain, i)], mapping[(chain, res)]]
                        row_values[j][k] = f"{pae_value:.2f}"
                        row_values_mirror[k][j] = f"{pae_value_mirror:.2f}"
            for j, (chain, res) in enumerate(stretch):
                aa = residue_to_aa_mapping[(chain, res)]
                file.write(f"{chain}-{aa}{res}: " + " ".join(row_values[j]))
                if (structured_chain, 62+j) in mapping and 62+j<70:
                    saa = residue_to_aa_mapping[(structured_chain, 62+j)]
                    file.write(" "*3 + f"{structured_chain}-{saa}{62+j}: " + " ".join(row_values_mirror[j]) + "\n")
            if (70-62>len(stretch)):
                for l, k in enumerate(range(62+len(stretch),70)):
                    saa = residue_to_aa_mapping[(structured_chain, k)]
                    file.write(" "*49 + f"{structured_chain}-{saa}{k}: " + " ".join(row_values_mirror[len(stretch)+l]) + "\n")
            elif(70-62<len(stretch)):
                file.write("\n")

            # Average PAE for the range 62 to 88
            for chain, res in stretch:
                for i in range(62, 89):
                    if (structured_chain, i) in mapping:
                        pae_value = paes[mapping[(chain, res)], mapping[(structured_chain, i)]]
                        pae_value_mirror = paes[mapping[(structured_chain, i)], mapping[(chain, res)]]
                        total_pae_62_88 += pae_value
                        total_pae_62_88_mirror += pae_value_mirror
                        count_62_88 += 1
                        count_62_88_mirror += 1
            avg_pae_62_88 = total_pae_62_88 / count_62_88 if count_62_88 > 0 else float('nan')
            avg_pae_62_88_mirror = total_pae_62_88_mirror / count_62_88_mirror if count_62_88_mirror > 0 else float('nan')
            file.write(f"Average PAE (62-88):: P->L {avg_pae_62_88:.3f} and L->P {avg_pae_62_88_mirror:.3f}\n")

            # Average PAE for the entire structured chain
            for chain, res in stretch:
                for i in range(1, max(mapping[(structured_chain, res)] for (structured_chain, res) in mapping if structured_chain == structured_chain) + 1):
                    if (structured_chain, i) in mapping:
                        pae_value = paes[mapping[(chain, res)], mapping[(structured_chain, i)]]
                        pae_value_mirror = paes[mapping[(structured_chain, i)], mapping[(chain, res)]]
                        total_pae_entire += pae_value
                        total_pae_entire_mirror += pae_value_mirror
                        count_entire += 1
                        count_entire_mirror += 1
            avg_pae_entire = total_pae_entire / count_entire if count_entire > 0 else float('nan')
            avg_pae_entire_mirror = total_pae_entire_mirror / count_entire_mirror if count_entire_mirror > 0 else float('nan')

            file.write(f"Average PAE (entire LC8 protomer):: P->L: {avg_pae_entire:.3f} and L->P: {avg_pae_entire_mirror:.3f}\n\n\n")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python analyze_structures.py fasta_file pdb_file json_file output_directory rank_part")
        sys.exit(1)

    fasta_file = sys.argv[1]
    pdb_file = sys.argv[2]
    json_file = sys.argv[3]
    output_directory = sys.argv[4]
    rank_part = sys.argv[5]

    sequences = read_fasta(fasta_file)
    ca_coords = parse_pdb(pdb_file)
    #print(sequences)
    close_residues = get_close_residues(ca_coords)
    plddts, paes, conf = analyze_json(json_file)
    residue_index_mapping = create_residue_index_mapping(sequences)
    residue_to_aa_mapping = create_residue_to_aa_mapping(sequences)
    #print(residue_index_mapping)

    structured_chains = 'CDEFGHIJKLMNOPQRSTUVWXYZ'
    binding_regions = find_binding_regions(ca_coords, close_residues, structured_chains)
    
    output_filename = f"{output_directory}/{rank_part}_interaction_analysis.txt"
    write_output(output_filename, close_residues, paes, residue_index_mapping, binding_regions, residue_to_aa_mapping, plddts, conf)

    print(f"Analysis complete for {rank_part}. Output written.")
