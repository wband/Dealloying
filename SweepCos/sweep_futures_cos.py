import sys
import csv
import itertools
import os
import subprocess
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Central Configuration ---
SWEEP_PARAMETERS = {
 #   "D2": [500.0],
 #   "DELTAG0": [1.0],
 #   "X1": [0.6, 0.2],
 #   "X2": [0.01],
    "L_INT": [1.5, 1.0, 0.5],
    "D1S": [0.0, 1.0, 10.0],
    "DEL": [0, 1, 2],
    "A": [0.5, 0.25]
    # "NEW_PARAM": [100, 200]  <-- Adding a new parameter is now this easy!
}

# parameter name (for table) and index (in simulation) and whether to normalize the integrated value by LX
OUTPUT_VARIABLES = {
    "phase_field": (0, True),
    "interfacial_energy": (6, False),
    "Ni_fraction": (7, True),
}

N_PROCESSES = 4
TIME_SCALE = 1.0
MAX_ITERS = 1000000
INITIAL_DT = 2.048e-4
MIN_DT = 1e-9
DT_REDUCTION_FACTOR = 0.5

TEMPLATE_PRM = "base_parameters.prm"
BASE_OUTPUT_DIR = "sim_outputs"
EXE_PATH = os.path.abspath("build/main")
LOG_FILE = "sweep_results.csv"

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

    # Set the final time step based on the length scale and the initial dt
    end_increment = min(int(TIME_SCALE/dt*param_dict["L_INT"]**2), MAX_ITERS)
    content = content.replace("__MAXITS__", str(end_increment))

    # set the length scale based on the L_INT parameter
    Lx = param_dict["L_INT"]*64.0/10.0
    content = content.replace("__LX__", str(Lx))
   
    with open(output_path, 'w') as f:
        f.write(content)

def run_single_combination(param_dict):
    """
    Handles the stability retry loop for ONE specific parameter combination.
    Runs locally on a single core inside the assigned process worker.
    """
    dt = INITIAL_DT
    attempts = 0
    success = False
    final_dir = ""
    status_msg = ""

    # Build unique directory name for this combo
    dir_parts = [f"{key}_{val}" for key, val in param_dict.items()]
    run_dir_name = "run_" + "_".join(dir_parts)

    while not success and dt >= MIN_DT:
        attempts += 1
        attempt_dir = os.path.join(BASE_OUTPUT_DIR, run_dir_name, f"attempt_{attempts}")
        os.makedirs(attempt_dir, exist_ok=True)
        
        prm_path = os.path.join(attempt_dir, "parameters.prm")
        generate_prm_file(TEMPLATE_PRM, prm_path, param_dict, dt)
        
        # Execute binary sequentially on its allocated core
        cmd = ["mpirun", "-n", "4", EXE_PATH, "-i", "parameters.prm"]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=attempt_dir, timeout=3600)
            with open(os.path.join(attempt_dir, "stdout.log"), "w") as f:
                f.write(result.stdout)
            success = True
            status_msg = "Success"
            final_dir = attempt_dir
        except subprocess.TimeoutExpired:
            status_msg = "Timeout"
            dt *= DT_REDUCTION_FACTOR
        except subprocess.CalledProcessError as e:
            with open(os.path.join(attempt_dir, "crash_stderr.log"), "w") as f:
                f.write(e.stderr)
            status_msg = f"Crash ({e.returncode})"
            dt *= DT_REDUCTION_FACTOR

    if not success:
        status_msg = "Failed (Min DT reached)"
        final_dt = "N/A"
    else:
        final_dt = dt

    # Extract output variables
    output_values = []
    for output_name, pp_props in OUTPUT_VARIABLES.items():
        log_file_path = os.path.join(final_dir, "summary.log")
        solution_index, is_normalized = pp_props
        last_solution = get_last_solution_index(log_file_path, solution_index)
        if last_solution:
            if is_normalized:
                output_values.append(last_solution["integrated_value"] / (param_dict["L_INT"]*128.0/10.0))
            else:
                output_values.append(last_solution["integrated_value"])
        else:
            output_values.append("N/A")  # Handle missing data gracefully}}}

    # Return data payload back to the main aggregator loop
    return param_dict, final_dt, status_msg, attempts, final_dir, output_values

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

def main():
    # 1. Generate all combinations globally
    param_names = list(SWEEP_PARAMETERS.keys())
    value_lists = list(SWEEP_PARAMETERS.values())
    output_names = list(OUTPUT_VARIABLES.keys())
    all_combinations = [dict(zip(param_names, combo)) for combo in itertools.product(*value_lists)]
    
    # 2. Get SLURM Environment variables
    # If running locally for testing, default to array index 0 and 1 total chunk
    array_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    total_chunks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    
    # Get available CPU cores assigned to this SLURM task
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", N_PROCESSES)) 
    
    # 3. Slice the master list for *this* array task
    avg = len(all_combinations) / float(total_chunks)
    start = int(avg * array_idx)
    end = int(avg * (array_idx + 1))
    my_tasks = all_combinations[start:end]
    
    log_file = f"sweep_results_array_{array_idx}.csv"
    print(f"Array Job {array_idx}: Processing {len(my_tasks)} combinations using {num_workers} local workers.")

    # 4. Process tasks using a greedy local worker pool
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(param_names + ["final_dt", "status", "attempts", "output_directory"]
                            + output_names)
        
        # ProcessPoolExecutor naturally load balances across the allocated cores
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Map tasks to the executor
            futures = {executor.submit(run_single_combination, task): task for task in my_tasks}
            
            # As soon as ANY single core finishes, grab its result immediately (as_completed)
            for future in as_completed(futures):
                param_dict, final_dt, status_msg, attempts, final_dir, output_values = future.result()
                
                # Write to this array slot's log file right away
                row_data = [param_dict[name] for name in param_names] \
                   + [final_dt, status_msg, attempts, final_dir] + output_values
                writer.writerow(row_data)
                f.flush()


if __name__ == "__main__":
    main()