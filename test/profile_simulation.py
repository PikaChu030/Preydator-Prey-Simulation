import cProfile
import subprocess
import pstats
from line_profiler import LineProfiler
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from radon.complexity import cc_visit
import inspect
from memory_profiler import memory_usage, profile
import memray
import numba

# Add the predator_prey directory to sys.path for module import
sys.path.append(os.path.abspath('./predator_prey'))

# Import the simulate_predator_prey module
import simulate_predator_prey as spp

def cprofile_simulation():
    # Analyze source code by cProfule
    subprocess.run(['python', '-m', 'cProfile', '-o', 'cprofile_results.prof', './predator_prey/simulate_predator_prey.py', '-f', './landscapes/test.dat'])
    
    # Print out the profiling stats for analysis
    profile_path = './cprofile_results.prof'
    stats = pstats.Stats(profile_path)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)
    stats.strip_dirs().sort_stats('time').print_stats(10)

def line_profile_simulation(func):
    # Check if the function is compiled with Numba
    if isinstance(func, numba.core.registry.CPUDispatcher):
        print(f"Function {func.__name__} is compiled with Numba. Skipping line profiling.")
        return  # Skip line profiling for Numba functions
        
    # Check if the function has been wrapped (common with decorators like njit)
    if hasattr(func, '__wrapped__'):
        func = func.__wrapped__  # Access the original function

    # Set up the line profiler
    profiler = LineProfiler()
    # Decorate the original (unwrapped) function to profile
    profiler.add_function(func)

    # Wrap simulation directly to profile calls to the specified function
    def simulation_with_profiling():
        # Parameters for the simulation
        mouse_params = spp.BiomeParameters(0.1, 0.05, 0.2)
        fox_params = spp.BiomeParameters(0.03, 0.09, 0.2)
        delta_t = 0.5
        time_step_interval = 10
        duration = 500
        landscape_file = './landscapes/test.dat'
        mouse_seed = 1
        fox_seed = 1

        # Run the simulation
        spp.simulate(mouse_params, fox_params, delta_t, time_step_interval,
                     duration, landscape_file, mouse_seed, fox_seed)

    # Profile the entire simulation code but focus on the specified function
    profiler_wrapper = profiler(simulation_with_profiling)
    profiler_wrapper()

    # Dump and print stats
    profiler.dump_stats(f'line_profiler_{func.__name__}.lprof')
    profiler.print_stats()

    # Visualize the results
    visualize_line_profiler(profiler.get_stats(), func.__name__)

def visualize_line_profiler(lp_stats, func_name):
    # Process the line profile data
    key = next(iter(lp_stats.timings.keys()))  # Get the first function key
    line_timing = lp_stats.timings[key]
    lines, hits, times = zip(*[(line[0], line[1], line[2]) for line in line_timing])

    plt.figure(figsize=(14, 5))
    plt.bar(lines, times, color='skyblue', alpha=0.7, edgecolor='black')
    plt.title(f'Line-by-Line Execution Time for {func_name}')
    plt.xlabel('Line Number')
    plt.ylabel('Execution Time (microseconds)')
    plt.savefig(f'line_profiler_{func_name}.png')
    plt.close()

def memory_profiling():
    # Define the simulation parameters
    mouse_params = spp.BiomeParameters(0.1, 0.05, 0.2)
    fox_params = spp.BiomeParameters(0.03, 0.09, 0.2)
    delta_t = 0.5
    time_step_interval = 10
    duration = 500
    landscape_file = './landscapes/test.dat'
    mouse_seed = 1
    fox_seed = 1

    def run_simulation():
        spp.simulate(mouse_params, fox_params, delta_t, time_step_interval,
                     duration, landscape_file, mouse_seed, fox_seed)

    # Measure memory usage
    mem_usage = memory_usage(run_simulation, interval=0.1, timeout=None)
    
    # Peak memory usage
    peak_memory = max(mem_usage)
    print(f"Peak Memory Usage: {peak_memory} MiB")
    
    # Plot memory usage over time
    plt.figure(figsize=(14, 5))
    plt.plot(mem_usage)
    plt.title('Memory Usage Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (MiB)')
    plt.savefig('memory_usage_plot.png')
    plt.close()
    
    # Measure memory allocation with memray
    with memray.Tracker("memray_report.bin"):
        run_simulation()
        
    # Generate memray flamegraph, summary and table reports
    subprocess.run(['python', '-m', 'memray', 'flamegraph', 'memray_report.bin', '-o', 'flamegraph.html'])
    subprocess.run('python -m memray summary memray_report.bin > summary.html', shell=True)
    subprocess.run(['python', '-m', 'memray', 'table', 'memray_report.bin', '-o', 'table.html'])

def analyze_function_complexity(function):
    '''
    Analyze and print the cyclomatic complexity of a given function.

    Parameters:
    function (function): The function to analyze.

    Returns:
    Tuple containing function name and its cyclomatic complexity.
    '''
    # Get the source code of the function
    source = inspect.getsource(function)
    # Analyze the source code to get cyclomatic complexity
    complexities = cc_visit(source)
    
    if complexities:
        return complexities[0].name, complexities[0].complexity
    else:
        return function.__name__, None

def analyze_complexity():
    # List of functions to analyze
    functions_to_analyze = [
        spp.check_positive_float,
        spp.check_positive_int,
        spp.parse_arguments,
        spp.read_landscape,
        spp.initialize_density,
        spp.calculate_land_neighbors,
        spp.calculate_average_density,
        spp.update_densities,
        spp.generate_color_maps,
        spp.write_ppm_file,
        spp.simulate
    ]

    # Lists to store results for visualization
    function_names = []
    complexities = []

    # Analyze and print cyclomatic complexity for each function
    print("\nFunction Cyclomatic Complexity Analysis:")
    for func in functions_to_analyze:
        func_name, complexity = analyze_function_complexity(func)
        if complexity is not None:
            print(f"Function {func_name}: Cyclomatic Complexity = {complexity}")
            function_names.append(func_name)
            complexities.append(complexity)
        else:
            print(f"Function {func_name}: Complexity could not be determined")

    # Visualize results
    visualize_complexity(function_names, complexities)

def visualize_complexity(function_names, complexities):
    plt.figure(figsize=(14, 5))
 
    plt.plot(function_names, complexities, marker='o', linestyle='-', color='b', label='Cyclomatic Complexity')
    plt.xticks(rotation=15, ha='right')

    plt.title('Cyclomatic Complexity of Functions')
    plt.xlabel('Function')
    plt.ylabel('Cyclomatic Complexity')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig('complexity_analysis.png')
    plt.close()

if __name__ == '__main__':
    print("Running cProfile...")
    cprofile_simulation()

    print("\nRunning line_profiler for update densities function...")
    line_profile_simulation(spp.update_densities)
    
    print("\nRunning memory_profiler...")
    memory_profiling()
    
    print("\nRunning Complexity Analysis...")
    analyze_complexity()
