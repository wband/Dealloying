import os
import re
import shutil
from pathlib import Path
import math

import os
import re
import shutil
import bisect
from pathlib import Path

def get_all_vtu_times(run_sequence):
    """
    Scans all sequential runs and returns a list of tuples:
    [(file_path, calculated_sim_time), ...] sorted chronologically.
    """
    pattern = re.compile(r'^solution[_-](\d+)(?:\.vtu)?$', re.IGNORECASE)
    all_files = []
    accumulated_time = 0.0
    max_time = 300.0
    for folder_path, dt in run_sequence:
        top_path = Path(folder_path)
        if not top_path.exists():
            print(f"Warning: Directory '{folder_path}' does not exist. Skipping.")
            continue

        run_files = []
        for file_path in top_path.rglob("solution*"):
            if file_path.is_file():
                match = pattern.match(file_path.name)
                if match:
                    iteration = int(match.group(1))
                    run_files.append((iteration, file_path))

        if not run_files:
            continue

        run_files.sort(key=lambda x: x[0])
        first_iter = run_files[0][0]
        max_iter_in_run = run_files[-1][0]

        for iteration, file_path in run_files:
            file_sim_time = accumulated_time + (iteration - first_iter) * dt
            if file_sim_time > max_time:
                break
            all_files.append((file_path, file_sim_time))

        # Update base time offset for the next sequential run
        run_duration = (max_iter_in_run - first_iter) * dt
        accumulated_time += run_duration
        if  accumulated_time > max_time:
            break

    all_files.sort(key=lambda x: x[1])
    return all_files


def consolidate_vtu_files(run_sequence, output_dir, target_dt, time_precision=6, allow_duplicates=False):
    """
    Copies and renames VTU files to a destination directory at uniform time increments.

    :param run_sequence: Chronological list of (folder_path, dt) tuples.
    :param output_dir: Destination path.
    :param target_dt: Physical simulation time increment.
    :param time_precision: Number of decimal places in output filenames.
    :param allow_duplicates: If False, skips copying the exact same source file more than once.
    """
    all_files = get_all_vtu_times(run_sequence)
    if not all_files:
        print("No solution files found in the provided sequence.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    times = [x[1] for x in all_files]
    first_time = times[0]
    last_time = times[-1]
    
    # Calculate zero-padding digits based on maximum whole integer part of last_time
    max_integer_val = max(1, int(math.floor(last_time)))
    int_digits = len(str(max_integer_val))

    # Generate expected target times
    target_times = []
    curr_t = first_time
    while curr_t <= last_time + (target_dt * 1e-6):
        target_times.append(curr_t)
        curr_t += target_dt

    copied_count = 0
    seen_source_files = set()

    print(f"Consolidating files to: {out_path.resolve()}")
    print(f"Total available source files: {len(all_files)}")
    print(f"Sampling target dt: {target_dt}")
    print("-" * 60)

    for target_t in target_times:
        # Binary search for closest time in ordered sequence
        pos = bisect.bisect_left(times, target_t)
        
        # Check surrounding indices to find the absolute minimum distance
        candidates = []
        for p in (pos - 1, pos, pos + 1):
            if 0 <= p < len(times):
                candidates.append((abs(times[p] - target_t), p))
        
        best_idx = min(candidates, key=lambda x: x[0])[1]
        best_file, best_time = all_files[best_idx]

        # Prevent duplicate copying of the same underlying file unless explicitly requested
        if not allow_duplicates and best_file in seen_source_files:
            continue
        seen_source_files.add(best_file)

        # Format new filename using actual simulation time
        whole_part = int(best_time)
        frac_part = abs(best_time - whole_part)
        frac_str = f"{frac_part:.{time_precision}f}".split('.')[1]
        time_str = f"{whole_part:0{int_digits}d}{frac_str}"
        new_filename = f"solution_t{time_str}.vtu"
        dest_file_path = out_path / new_filename

        shutil.copy2(best_file, dest_file_path)
        print(f"Target t={target_t:10.4f} -> Copied: {best_file.name} (sim t={best_time:.6f}) as {new_filename}")
        copied_count += 1

    print("-" * 60)
    print(f"Done! Successfully copied {copied_count} distinct file(s) to '{output_dir}'.")

if __name__ == "__main__":
    run_sequence_list = []
    run_sequence_list.append([
        ("./dG0_D250.0_D1S0.0/old/solutions_r1", 1.0e-5),
        ("./dG0_D250.0_D1S0.0/old/solutions_r2", 1.0e-5),
        ("./dG0_D250.0_D1S0.0/old/solutions_r3", 1.0e-5),
        ("./dG0_D250.0_D1S0.0/old2/solutions_r1", 1.0e-5),
        ("./dG0_D250.0_D1S0.0/old2/solutions_r2", 1.0e-5),
        ("./dG0_D250.0_D1S0.0/solutions_r1", 1.0e-5),
        ("./dG0_D250.0_D1S0.0/solutions_r2", 1.0e-5),
    ])

    run_sequence_list.append([
        ("./dG1_D210.0_D1S0.0/old/solutions_r1", 1.0e-5),
        ("./dG1_D210.0_D1S0.0/old/solutions_r2", 1.0e-5),
        ("./dG1_D210.0_D1S0.0/old/solutions_r3", 1.0e-5),
        ("./dG1_D210.0_D1S0.0/old2/solutions_r1", 1.0e-5),
        ("./dG1_D210.0_D1S0.0/old2/solutions_r2", 1.0e-5),
    ])
    run_sequence_list.append([
        ("./dG1_D250.0_D1S0.0/unstable1/solutions_r1", 1.0e-5),
        ("./dG1_D250.0_D1S0.0/unstable1/solutions_r2", 1.0e-5),
        ("./dG1_D250.0_D1S0.0/old2/solutions_r1", 5.0e-6),
        ("./dG1_D250.0_D1S0.0/old2/solutions_r2", 5.0e-6),
        ("./dG1_D250.0_D1S0.0/solutions_r1", 5.0e-6),
        ("./dG1_D250.0_D1S0.0/solutions_r2", 5.0e-6),
    ])
    
    run_sequence_list.append([
        ("./dG1_D250.0_D1S5.0/unstable1", 1.0e-5),
        ("./dG1_D250.0_D1S5.0/old2/solutions_r1", 5.0e-6),
        ("./dG1_D250.0_D1S5.0/old2/solutions_r2", 5.0e-6),
        ("./dG1_D250.0_D1S5.0/solutions_r1", 5.0e-6),
        ("./dG1_D250.0_D1S5.0/solutions_r2", 5.0e-6),
        ("./dG1_D250.0_D1S5.0/solutions", 5.0e-6),
    ])

    OUTPUT_DIRECTORY = "./small_base"
    TARGET_DT = 10.0  # Save a file every 0.05 physical time units

    consolidate_vtu_files(run_sequence_list[0], OUTPUT_DIRECTORY, TARGET_DT, time_precision=1)
