import sys
import csv
import itertools
import os
import subprocess
import re

# ==============================================================================
# 1. CENTRAL CONFIGURATION: Add or remove any parameters right here!
# ==============================================================================
SWEEP_PARAMETERS = {
    "D2": [10.0, 50.0, 100.0, 500.0],
    "DELTAG0": [2.0, 1.0, 0.5, 0.0],
    "X1": [0.6, 0.2],
    "X2": [0.2, 0.1, 0.05, 0.01],
    "L_INT": [2.0, 1.5, 1.0, 0.75],
    "D1S": [0.0, 1.0, 5.0],
    # "NEW_PARAM": [100, 200]  <-- Adding a new parameter is now this easy!
}

# parameter name (for table) and index (in simulation)
OUTPUT_VARIABLES = {
    "phase_field": 0,
    "interfacial_energy": 6,
    "Ni_fraction": 7,
}

INITIAL_DT = 2.048e-4
MIN_DT = 1e-10
DT_REDUCTION_FACTOR = 0.5

TEMPLATE_PRM = "base_parameters.prm"
BASE_OUTPUT_DIR = "sim_outputs"
EXE_PATH = os.path.abspath("build/main")
# ==============================================================================

def generate_prm_file(template_path, output_path, param_dict, dt):
    """
    Dynamically replaces placeholders in the template file.
    Looks for tags like __D2__, __DELTAG0__, etc., based on SWEEP_PARAMETERS keys.
    """
    with open(template_path, 'r') as f:
        content = f.read()

    # Automatically loop through whatever parameters exist in our config dict
    for key, value in param_dict.items():
        placeholder = f"__{key}__"
        content = content.replace(placeholder, str(value))
    
    # Always replace the time step
    content = content.replace("__DT__", str(dt))

    # Replace the length scale as well
    content = content.replace("__LX__", str(Lx))
    Lx = param_dict["L_INT"]*128.0/10.0

   
    with open(output_path, 'w') as f:
        f.write(content)


def run_simulation(param_dict, dt, attempt):
    """
    Creates isolated directories and runs the simulation using 'cwd'.
    """
    # Dynamically build folder name: e.g., "run_D2_0.1_DELTAG0_-5.0..."
    dir_parts = [f"{key}_{val}" for key, val in param_dict.items()]
    run_dir_name = "run_" + "_".join(dir_parts)
    
    attempt_dir = os.path.join(BASE_OUTPUT_DIR, run_dir_name, f"attempt_{attempt}")
    os.makedirs(attempt_dir, exist_ok=True)
    
    prm_path = os.path.join(attempt_dir, "parameters.prm")
    generate_prm_file(TEMPLATE_PRM, prm_path, param_dict, dt)
    
    cmd = ["mpirun", "-n", "2", EXE_PATH, "-i", "parameters.prm"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=attempt_dir)
        with open(os.path.join(attempt_dir, "stdout.log"), "w") as f:
            f.write(result.stdout)
        return True, "Success", attempt_dir
    except subprocess.CalledProcessError as e:
        with open(os.path.join(attempt_dir, "crash_stderr.log"), "w") as f:
            f.write(e.stderr)
        return False, f"Crash (Exit code {e.returncode})", attempt_dir

def get_last_solution_index(file_path, solution_index):
    """
    Scans a text file and returns the l2-norm and integrated value 
    for a specific solution index from the very last iteration block.
    """
    last_values = None
    
    # Dynamically inject the solution index into the regex pattern
    # \s+ handles normal or non-breaking spaces safely
    pattern = re.compile(
        r"Solution\s+index\s+" + str(solution_index) + r"\s+l2-norm:\s*([\d\.\-]+)\s+integrated\s+value:\s*([\d\.\-]+)"
    )
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # Continuously overwrite so we are left with the final iteration's data
                l2_norm = float(match.group(1))
                integrated_value = float(match.group(2))
                last_values = {
                    "solution_index": solution_index,
                    "l2_norm": l2_norm,
                    "integrated_value": integrated_value
                }
    return last_values

# --- Example Usage ---
# file_name = 'log_output.txt'
# print(get_last_solution_index(file_name, 6))
# print(get_last_solution_index(file_name, 3))
# print(get_last_solution_index(file_name, 99)) # Will return None safely


def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Extract keys and value lists in a guaranteed matching order
    param_names = list(SWEEP_PARAMETERS.keys())
    output_names = list(OUTPUT_VARIABLES.keys())
    value_lists = list(SWEEP_PARAMETERS.values())
    
    # Generate Cartesian product
    raw_combinations = list(itertools.product(*value_lists))
    
    # Parallelization chunking
    if len(sys.argv) == 3:
        chunk_idx = int(sys.argv[1])
        total_chunks = int(sys.argv[2])
        avg = len(raw_combinations) / float(total_chunks)
        start, end = int(avg * chunk_idx), int(avg * (chunk_idx + 1))
        my_raw_combos = raw_combinations[start:end]
        my_log_file = f"sweep_results_chunk_{chunk_idx}.csv"
    else:
        my_raw_combos = raw_combinations
        my_log_file = "sweep_results.csv"

    file_exists = os.path.isfile(my_log_file)
    with open(my_log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Dynamically write CSV headers based on your parameter names
        if not file_exists:
            headers = param_names + ["final_dt", "status", "attempts", "output_directory"] + output_names
            writer.writerow(headers)

        for i, combo in enumerate(my_raw_combos, 1):
            # Zip the names and specific values back into a neat dictionary for this run
            # e.g., {"D2": 0.1, "DELTAG0": -5.0, ...}
            current_params = dict(zip(param_names, combo))
            
            # Print status summary
            param_str = ", ".join([f"{k}={v}" for k, v in current_params.items()])
            print(f"\n[Combo {i}/{len(my_raw_combos)}] Testing: {param_str}")
            
            dt = INITIAL_DT
            attempts = 0
            success = False
            final_dir = ""

            while not success and dt >= MIN_DT:
                attempts += 1
                success, status_msg, attempt_dir = run_simulation(current_params, dt, attempts)
                final_dir = attempt_dir
                
                if success:
                    print(f"  ✅ Success! Saved in: {attempt_dir}")
                    break
                else:
                    print(f"  ❌ Failed (dt={dt}). Retrying...")
                    dt *= DT_REDUCTION_FACTOR

            # Extract output variables
            output_values = []
            for output_name, solution_index in OUTPUT_VARIABLES.items():
                log_file_path = os.path.join(final_dir, "summary.log")
                last_solution = get_last_solution_index(log_file_path, solution_index)
                if last_solution:
                    output_values.append(last_solution["integrated_value"])
                else:
                    output_values.append("N/A")  # Handle missing data gracefully

            # Dynamically build the log row
            row_data = [current_params[name] for name in param_names] + [
                dt if success else "N/A", 
                status_msg, 
                attempts, 
                final_dir
            ] + output_values
            writer.writerow(row_data)
            f.flush()

if __name__ == "__main__":
    main()