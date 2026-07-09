import sys
import csv
import itertools
import os
import subprocess
import re
from mpi4py import MPI

# ==============================================================================
# 1. CENTRAL CONFIGURATION: Add or remove any parameters right here!
# ==============================================================================
SWEEP_PARAMETERS = {
    "D2": [50.0],
    "DELTAG0": [1.0, 0.0],
    "X1": [0.2],
    "X2": [0.05],
    "L_INT": [1.5, 1.0],
    "D1S": [1.0, 0.0],
    "DEL": [0, 1, 2],
    # "NEW_PARAM": [100, 200]  <-- Adding a new parameter is now this easy!
}

# parameter name (for table) and index (in simulation) and whether to normalize the integrated value by LX
OUTPUT_VARIABLES = {
    "phase_field": (0, True),
    "interfacial_energy": (6, False),
    "Ni_fraction": (7, True),
}

TIME_SCALE = 1.0
MAX_ITERS = 16000000
INITIAL_DT = 2.048e-4
MIN_DT = 1e-10
DT_REDUCTION_FACTOR = 0.5

TEMPLATE_PRM = "base_parameters.prm"
BASE_OUTPUT_DIR = "sim_outputs"
EXE_PATH = os.path.abspath("build/main")
LOG_FILE = "sweep_results.csv"
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

    # Set the final time step based on the length scale and the initial dt
    end_increment = min(int(TIME_SCALE/dt*param_dict["L_INT"]**2), MAX_ITERS)
    content = content.replace("__MAXITS__", str(end_increment))

    # set the length scale based on the L_INT parameter
    Lx = param_dict["L_INT"]*128.0/10.0
    content = content.replace("__LX__", str(Lx))
   
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
    
    cmd = ["mpirun", "-n", "1", EXE_PATH, "-i", "parameters.prm"]
    
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

# MPI Tags to govern communication states
TAG_READY = 1
TAG_TASK = 2
TAG_DONE = 3
TAG_EXIT = 4

def manager_main(comm, param_names, raw_combinations, output_names):
    """
    Rank 0: Hand out tasks dynamically, collect results, and write to a single log.
    """
    size = comm.Get_size()
    total_tasks = len(raw_combinations)
    print(f"[Manager] Starting sweep with {total_tasks} tasks across {size} MPI ranks.")
    
    # Initialize the single master CSV file
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(param_names + ["final_dt", "status", "attempts", "output_directory"]
                            + output_names)

        task_idx = 0
        active_workers = size - 1

        # While there are tasks to give out or workers still computing
        while active_workers > 0:
            status = MPI.Status()
            # Wait for any worker to send data/request
            incoming_data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker_rank = status.Get_source()
            tag = status.Get_tag()

            if tag == TAG_DONE:
                # A worker finished a simulation. Unpack and log the data.
                current_params, dt, status_msg, attempts, final_dir, output_values = incoming_data
                row_data = [current_params[name] for name in param_names] \
                    + [dt, status_msg, attempts, final_dir] \
                    + output_values
                writer.writerow(row_data)
                f.flush()
                print(f"[Manager] Saved result from Rank {worker_rank}. ({task_idx}/{total_tasks} launched)")

            if task_idx < total_tasks:
                # If there are tasks left, send the next combo to the requesting worker
                next_combo = raw_combinations[task_idx]
                current_params = dict(zip(param_names, next_combo))
                comm.send(current_params, dest=worker_rank, tag=TAG_TASK)
                task_idx += 1
            else:
                # No tasks left, tell the worker it's time to shut down
                comm.send(None, dest=worker_rank, tag=TAG_EXIT)
                active_workers -= 1

    print("[Manager] Master sweep complete. All workers cleanly exited.")


def worker_main(comm):
    """
    Ranks 1+: Request a task, run the execution loop, report back, repeat.
    """
    rank = comm.Get_rank()
    
    # Let the manager know this worker is alive and ready for its first task
    comm.send(None, dest=0, tag=TAG_READY)

    while True:
        status = MPI.Status()
        # Receive instructions from the manager
        task_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAG_EXIT:
            break  # Break out of loop to exit cleanly

        if tag == TAG_TASK:
            current_params = task_data
            
            # Run the inner simulation stability loop locally on this worker rank
            dt = INITIAL_DT
            attempts = 0
            success = False
            final_dir = ""
            status_msg = ""

            while not success and dt >= MIN_DT:
                attempts += 1
                success, status_msg, attempt_dir = run_simulation(current_params, dt, attempts)
                final_dir = attempt_dir
                
                if success:
                    break
                else:
                    dt *= DT_REDUCTION_FACTOR

            final_dt = dt if success else "N/A"
            if not success:
                status_msg = "Failed (Min DT reached)"

            # Extract output variables
            output_values = []
            for output_name, pp_props in OUTPUT_VARIABLES.items():
                log_file_path = os.path.join(final_dir, "summary.log")
                solution_index, is_normalized = pp_props
                last_solution = get_last_solution_index(log_file_path, solution_index)
                if last_solution:
                    if is_normalized:
                        output_values.append(last_solution["integrated_value"] / (current_params["L_INT"]*128.0/10.0))
                    else:
                        output_values.append(last_solution["integrated_value"])
                else:
                    output_values.append("N/A")  # Handle missing data gracefully

            # Pack up the results and send them back to the manager
            result_payload = (current_params, final_dt, status_msg, attempts, final_dir, output_values)
            comm.send(result_payload, dest=0, tag=TAG_DONE)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Extract configuration properties
    param_names = list(SWEEP_PARAMETERS.keys())
    value_lists = list(SWEEP_PARAMETERS.values())
    raw_combinations = list(itertools.product(*value_lists))
    output_names = list(OUTPUT_VARIABLES.keys())

    if rank == 0:
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        manager_main(comm, param_names, raw_combinations, output_names)
    else:
        worker_main(comm)


if __name__ == "__main__":
    main()