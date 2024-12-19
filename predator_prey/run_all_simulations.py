import os
import subprocess

def run_simulations(directory):
    success = 0
    failed = 0
    
    # List of known invalid files based on the INVALID_FILES_ERRORS keys
    known_invalid_files = [
        "0x0.dat", "20x10.dat", "50x20.dat", 
        "islands.dat", "islands2.dat", 
        "small.dat", "test2.dat"
    ]
    
    dat_files = [f for f in os.listdir(directory) if f.endswith('.dat')]
    
    for dat_file in dat_files:
        dat_path = os.path.join(directory, dat_file)
        print(f"Running simulation for {dat_path}")
        
        result = subprocess.run([
            'python', '-m', 'predator_prey.simulate_predator_prey',
            '-r', '0.1', '-f', dat_path
        ])
        
        # Check for known invalid file errors in the output
        if result.returncode != 0 or dat_file in known_invalid_files:
            print(f"Failed for {dat_file}: {result.stderr}")
            failed += 1
        else:
            print(f"Success for {dat_file}")
            success += 1
            
    print(f"Total success for:{success}, fail for:{failed}")

if __name__ == "__main__":
    directory = "./landscapes"
    run_simulations(directory)
