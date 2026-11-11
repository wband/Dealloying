import csv
import itertools
import re
from pathlib import Path
from sweep_futures import SWEEP_PARAMETERS, OUTPUT_VARIABLES

def get_run_metrics(param_dict: dict, base_output_dir: str | Path) -> dict:
    run_dict = param_dict.copy()  # Start with the parameters
    dir_parts = [f"{key}_{val}" for key, val in param_dict.items()]
    run_dir_name = Path("run_" + "_".join(dir_parts))
    final_dir = find_latest_attempt_folder(Path(base_output_dir) / run_dir_name)
    # Extract output variables
    output_values = []
    if (final_dir/"crash_stderr.log").exists():
        run_dict["status"] = "Crash"
        run_dict["final_dt"] = "N/A"
        run_dict = {**run_dict, **{key: "N/A" for key in OUTPUT_VARIABLES.keys()}}
    else:
        run_dict["status"] = "Success"
        run_dict["final_dt"] = extract_time_step(final_dir/"parameters.prm")
        for output_name, pp_props in OUTPUT_VARIABLES.items():
            log_file_path = final_dir/"summary.log"
            solution_index, is_normalized = pp_props
            last_solution = get_last_solution_index(log_file_path, solution_index)
            if last_solution:
                if is_normalized:
                    run_dict[output_name] = last_solution["integrated_value"] / (param_dict["L_INT"]*128.0/10.0)
                else:
                    run_dict[output_name] = last_solution["integrated_value"]
            else:
                run_dict[output_name] = "N/A"  # Handle missing data gracefully
    return run_dict


def extract_time_step(file_path: str | Path) -> float | None:
    # This regex matches "set time step =", ignores variable spaces, 
    # and captures integers, decimals, or scientific notation (like 1e-7)
    pattern = re.compile(
        r"set\s+time\s+step\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    )
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Convert the captured string (e.g., "1e-7") to a Python float
                return float(match.group("value")) 
    return None

def find_latest_attempt_folder(base_directory: str | Path, prefix: str = "attempt_") -> Path | None:
    """
    Finds the folder with the highest suffix number matching the pattern {prefix}{number}.
    
    Example:
        If the directory has 'attempt_1', 'attempt_2', 'attempt_10', 
        this returns the Path object for 'attempt_10'.
    """
    base_path = Path(base_directory)
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"The path '{base_directory}' is not a valid directory.")

    # Regex to match the prefix followed by one or more digits
    pattern = re.compile(f"^{re.escape(prefix)}(\\d+)$")
    matching_folders = []
    # Iterate through all items in the base directory
    for item in base_path.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                # Store a tuple of (integer_value, path_object)
                attempt_num = int(match.group(1))
                matching_folders.append((attempt_num, item))
                
    # If no matching folders are found, return None
    if not matching_folders:
        return None
        
    # Sort by the integer value (first item in the tuple) and return the Path of the highest
    matching_folders.sort(key=lambda x: x[0])
    return matching_folders[-1][1]

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
    param_names = list(SWEEP_PARAMETERS.keys())
    value_lists = list(SWEEP_PARAMETERS.values())
    output_names = list(OUTPUT_VARIABLES.keys())
    all_combinations = [dict(zip(param_names, combo)) for combo in itertools.product(*value_lists)]
    log_file = f"sweep_results_post.csv"
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(param_names + ["status", "final_dt"] + output_names)
        for param in all_combinations:
            out_dict = get_run_metrics(param, "sim_outputs")
            writer.writerow(out_dict.values())
            f.flush()

if __name__ == "__main__":
    main()