import csv
import itertools
import os
import sys
import subprocess

# intended invocation: python sweep.py [chunk_index] [total_chunks]

# --- Configuration ---
D2_vals = [100.0, 500.0]
deltaG0_vals = [1.0]
x1_vals = [0.6]
x2_vals = [0.01]
l_int_vals = [0.25, 0.5, 1.0]  # <--- Add your l_int values here

INITIAL_DT = 2.048e-4
MIN_DT = 1e-10
DT_REDUCTION_FACTOR = 0.5

TEMPLATE_PRM = "base_parameters.prm"
LOG_FILE = "sweep_results.csv"
BASE_OUTPUT_DIR = "sim_outputs"  # All simulation folders will go here
SIM_COMMAND = "build/main"   # Your Deal.II binary

def generate_prm_file(template_path, output_path,
                    D2, deltaG0, x1, x2, l_int, dt):
    Lx = l_int*128.0/10.0
    Ly = l_int*2.0/10.0
    with open(template_path, 'r') as f:
        content = f.read()

    content = content.replace("__LX__", str(Lx))
    content = content.replace("__LY__", str(Ly))
    content = content.replace("__DT__", str(dt))
    content = content.replace("__DELTAG0__", str(deltaG0))
    content = content.replace("__D2__", str(D2))
    content = content.replace("__L_INT__", str(l_int))
    content = content.replace("__X1__", str(x1))
    content = content.replace("__X2__", str(x2))

    with open(output_path, 'w') as f:
        f.write(content)

def run_simulation(D2, deltaG0, x1, x2, l_int, dt, run_id, attempt):
    # 1. Create the unique nested attempt directory
    run_dir_name = f"run_D{D2}_dG{deltaG0}_x1_{x1}_x2_{x2}_lint_{l_int}"
    attempt_dir = os.path.join(BASE_OUTPUT_DIR, run_dir_name, f"attempt_{attempt}")
    os.makedirs(attempt_dir, exist_ok=True)
    
    # 2. Get the ABSOLUTE path to your executable before we switch directories
    # (Assuming build/main is relative to where you start this python script)
    exe_path = os.path.abspath("build/main")
    
    # 3. Put the parameters.prm right inside the attempt directory
    prm_path = os.path.join(attempt_dir, "parameters.prm")
    
    # We no longer pass attempt_dir to this function since your .prm doesn't need it
    generate_prm_file(TEMPLATE_PRM, prm_path, D2, deltaG0, x1, x2, l_int, dt)
    
    # 4. Construct the command. 
    # Since we will execute *inside* attempt_dir, the parameter file is just "parameters.prm"
    cmd = ["mpirun", "-n", "1", exe_path, "-i", "parameters.prm"]
    
    try:
        # CRITICAL: cwd=attempt_dir forces mpirun and your sim to run inside that folder.
        # Your simulation will now create its 'solutions/' folder inside attempt_dir/
        result = subprocess.run(cmd, check=True,
            capture_output=True, text=True, cwd=attempt_dir)
        
        with open(os.path.join(attempt_dir, "stdout.log"), "w") as f:
            f.write(result.stdout)
            
        return True, "Success", attempt_dir
        
    except subprocess.CalledProcessError as e:
        with open(os.path.join(attempt_dir, "crash_stderr.log"), "w") as f:
            f.write(e.stderr)
        with open(os.path.join(attempt_dir, "crash_stdout.log"), "w") as f:
            f.write(e.stdout)
            
        return False, f"Crash (Exit code {e.returncode})", attempt_dir

def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # 1. Generate ALL possible combinations
    param_combinations = list(itertools.product(D2_vals, deltaG0_vals, x1_vals, x2_vals, l_int_vals))
    total_sims = len(param_combinations)

    # 2. Parse command-line arguments to split the workload
    # Usage: python script.py [chunk_index] [total_chunks]
    # Example for 4 shells: 
    # Shell 1: python script.py 0 4
    # Shell 2: python script.py 1 4  ...etc
    if len(sys.argv) == 3:
        chunk_idx = int(sys.argv[1])
        total_chunks = int(sys.argv[2])
        
        # Split the list into roughly equal segments
        avg = len(param_combinations) / float(total_chunks)
        start = int(avg * chunk_idx)
        end = int(avg * (chunk_idx + 1))
        
        my_combinations = param_combinations[start:end]
        # Unique log file per shell to prevent race conditions
        my_log_file = f"sweep_results_chunk_{chunk_idx}.csv"
        print(f"Running chunk {chunk_idx + 1}/{total_chunks}. Handling sims {start} to {end} ({len(my_combinations)} total).")
    else:
        # Default fallback: run everything if no args are passed
        my_combinations = param_combinations
        my_log_file = LOG_FILE
        print(f"Running full sweep ({total_sims} total combinations).")

    # 3. Process only this chunk's combinations
    file_exists = os.path.isfile(my_log_file)
    with open(my_log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["D2", "deltaG0", "x1", "x2", "l_int", "final_dt", "status", "attempts", "output_directory"])

        for i, (D2, deltaG0, x1, x2, l_int) in enumerate(my_combinations, 1):
            print(f"\n[Combo {i}/{len(my_combinations)}] Testing: D2={D2}, dG0={deltaG0}, x1={x1}, x2={x2}, l_int={l_int}")
            
            dt = INITIAL_DT
            attempts = 0
            success = False
            final_dir = ""

            while not success and dt >= MIN_DT:
                attempts += 1
                success, status_msg, attempt_dir = run_simulation(D2, deltaG0, x1, x2, l_int, dt, i, attempts)
                final_dir = attempt_dir
                
                if success:
                    print(f"  ✅ Success! Saved in: {attempt_dir}")
                    break
                else:
                    print(f"  ❌ Failed (dt={dt}). Retrying...")
                    dt *= DT_REDUCTION_FACTOR

            writer.writerow([D2, deltaG0, x1, x2, l_int, dt if success else "N/A", status_msg, attempts, final_dir])
            f.flush()


if __name__ == "__main__":
    main()