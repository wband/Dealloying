import sys
import csv

# invoke with /Applications/VisIt.app/Contents/Resources/bin/visit -cli -nowin -s extract-depth.py
# -----------------------------------------------------------------------------
# Configuration Parameters
# -----------------------------------------------------------------------------
database_path = "base_D1S_5/solution_t*.vtu database"  # Path to your data file or file series
expression_name = "yextent"       # Name of the custom expression
expression_defn = "(80 - coord(mesh)[1])*floor((1-n)/0.5)"   # Definition of your custom expression
output_file_path = "max_vs_time_D1S_5.csv"     # Output CSV file path
plot_var_type = "Pseudocolor"            # Plot type to display the expression

# -----------------------------------------------------------------------------
# Script Execution
# -----------------------------------------------------------------------------

if not OpenDatabase(database_path):
    print("Error: Could not open database.", file=sys.stderr)
    sys.exit(1)

# Define custom scalar expression
DefineScalarExpression(expression_name, expression_defn)

# Add plot and draw to enable query evaluation
AddPlot(plot_var_type, expression_name)
DrawPlots()

num_timesteps = TimeSliderGetNStates()

# Open CSV file for standard machine-readable output
with open(output_file_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    
    # Standard header row
    writer.writerow(["state", "time", "max_val"])
    
    for state in range(num_timesteps):
        TimeSliderSetState(state)
        
        # Query simulation time and maximum value
        Query("Time")
        current_time = float(GetQueryOutputValue())
        
        Query("Max")
        max_val = float(GetQueryOutputValue())
        
        # Write exact numeric records
        writer.writerow([state, current_time, max_val])

sys.exit(0)
