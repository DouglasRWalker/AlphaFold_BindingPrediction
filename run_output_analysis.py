import os
import subprocess
import re

def main():
    # Load necessary modules (if needed in your environment)
    # Example: os.system('module load pymol')

    # Iterate through the fasta files
    for fasta_file in os.listdir('.'):
        if fasta_file.endswith('.fasta'):
            base_name = f"{fasta_file[:-6]}_output"
            
            # Check if the directory exists
            if os.path.isdir(base_name):
                # Loop through rankx files within the directory
                for pdb_file in os.listdir(base_name):
                    if re.search(r'rank_.*\.pdb$', pdb_file):
                        pdb_file_path = os.path.join(base_name, pdb_file)
                        
                        # Define the corresponding json file name
                        rank_part_match = re.search(r'rank_[^_]*', pdb_file)
                        if rank_part_match:
                            rank_part = rank_part_match.group(0)
                            
                            for json_file in os.listdir(base_name):
                                if re.search(rf'{rank_part}.*\.json$', json_file):
                                    json_file_path = os.path.join(base_name, json_file)
                                    
                                    # Call the Python script with the necessary arguments
                                    subprocess.run(["python", "analyze_structures.py", fasta_file, pdb_file_path, json_file_path, base_name, rank_part])
                                #else:
                                    #print(f"Warning: Missing .json file for {pdb_file}")

                # Call organize_analysis.py script
                subprocess.run(["python", "organize_analysis.py", base_name])
            else:
                print(f"Warning: Directory {base_name} does not exist")

if __name__ == "__main__":
    main()
