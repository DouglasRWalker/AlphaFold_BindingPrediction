import sys
import os

def read_fasta(filename):
    """Reads a fasta file and returns the title and sequence."""
    with open(filename, 'r') as file:
        title = file.readline().strip()
        sequence = file.read().replace('\n', '')
    return title, sequence

def write_fasta(filename, title, sequence, parsed_filename):
    """Writes a fasta file with the given title and sequence."""
    n = 1#dictate the number of LC8 dimers you want to run with each sequence
    n = 2*n
    with open(filename, 'w') as file:
        file.write(f"{title}\n")
        file.write(f"{sequence}:\n")
        file.write(f"{sequence}:\n")
        LC8title,LC8sequence = read_fasta("LC8.fasta")
        for i in range(n-1):
            file.write(f"{LC8sequence}:\n")
        file.write(f"{LC8sequence}")
    with open(parsed_filename, 'a') as file2:
        file2.write(f"{sequence}\n")

def generate_fasta_files(input_filename):
    """Generates new fasta files with sequences of 16 amino acids starting at every 8th position."""
    title, sequence = read_fasta(input_filename)
    sequence_length = len(sequence)
    parsed_filename = f"{input_filename[:-6]}-parsed.fasta"
    
    with open(parsed_filename, 'w') as file2:
        file2.write(f"{title}_parsed\n")

    for i in range(0, sequence_length, 8):
        start = i
        end = i + 16
        if end > sequence_length:
            start = sequence_length - 16
            end = sequence_length
        subsequence = sequence[start:end]

        with open(parsed_filename, 'a') as file:
            file.write(f"{start}-{end}: ")
        
        output_filename = f"{input_filename[:-6]}_{start+1}-{end}.fasta"
        write_fasta(output_filename, title, subsequence, parsed_filename)

        if end == sequence_length:
            break

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_fasta.py fasta_file")
        sys.exit(1)

    input_basename = sys.argv[1]
    input_filename = f"{input_basename}.fasta"
    
    if not os.path.isfile(input_filename):
        print(f"Error: File '{input_filename}' not found.")
        sys.exit(1)

    generate_fasta_files(input_filename)
    print("Processing complete.")
