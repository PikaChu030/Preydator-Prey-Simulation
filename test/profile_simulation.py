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

# Add the predator_prey directory to sys.path for module import
sys.path.append(os.path.abspath('./predator_prey'))

# Import the simulate_predator_prey module
import simulate_predator_prey as spp

def cprofile_simulation():
    # Profile simulation using cProfile and visualize results
    profile = cProfile.Profile()
    profile.enable()

    # Parameters for the simulation
    mouse_params = spp.BiomeParameters(0.1, 0.05, 0.2)
    fox_params = spp.BiomeParameters(0.03, 0.09, 0.2)
    delta_t = 0.5
    time_step_interval = 10
    duration = 500

    # Path to the landscape file
    landscape_file = './landscapes/test.dat'
    mouse_seed = 1
    fox_seed = 1

    # Run the simulation
    spp.simulate(mouse_params, fox_params, delta_t, time_step_interval, duration, landscape_file, mouse_seed, fox_seed)

    profile.disable()

    # Output the profiling results as a .prof file
    profile_path = './cprofile_results.prof'
    profile.dump_stats(profile_path)

    # Print and visualize stats
    stats = pstats.Stats(profile_path)
    stats.sort_stats('cumulative').print_stats(10)
    stats.sort_stats('time').print_stats(10)

def line_profile_simulation():
    # Profile specific functions using line_profiler
    profiler = LineProfiler()
    profiler_wrapper = profiler(spp.simulate)

    # Parameters for the simulation
    mouse_params = spp.BiomeParameters(0.1, 0.05, 0.2)
    fox_params = spp.BiomeParameters(0.03, 0.09, 0.2)
    delta_t = 0.5
    time_step_interval = 10
    duration = 500
    landscape_file = './landscapes/test.dat'
    mouse_seed = 1
    fox_seed = 1

    profiler_wrapper(mouse_params, fox_params, delta_t, time_step_interval, duration, landscape_file, mouse_seed, fox_seed)
    profiler.print_stats()

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

    # Analyze and print cyclomatic complexity for each function
    print("\nFunction Cyclomatic Complexity Analysis:")
    for func in functions_to_analyze:
        func_name, complexity = analyze_function_complexity(func)
        if complexity is not None:
            print(f"Function {func_name}: Cyclomatic Complexity = {complexity}")
        else:
            print(f"Function {func_name}: Complexity could not be determined")


if __name__ == '__main__':
    print("Running cProfile...")
    cprofile_simulation()

    print("\nRunning line_profiler...")
    line_profile_simulation()

    print("\nRunning memory_profiler...")
    memory_profiling()
    
    print("\nRunning Complexity Analysis...")
    analyze_complexity()
