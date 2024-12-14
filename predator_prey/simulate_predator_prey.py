"""
Predator-prey simulation. Foxes and mice.

Version 3.0, last updated in December 2024.
"""

from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
import random
import os

def getVersion():
    """Returns the version of the simulation."""
    return 3.0

def check_positive_float(value):
    """Check if the argument is a positive float."""
    fvalue = float(value)
    if fvalue <= 0:
        raise ArgumentTypeError(f"{value} is not a positive float value")
    return fvalue

def check_positive_int(value):
    """Check if the argument is a positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        raise ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = ArgumentParser(description="Simulate predator-prey interactions between foxes and mice.")
    parser.add_argument("-r", "--birth-mice", type=check_positive_float, default=0.1, help="Birth rate of mice")
    parser.add_argument("-a", "--death-mice", type=check_positive_float, default=0.05, help="Rate at which foxes eat mice")
    parser.add_argument("-k", "--diffusion-mice", type=check_positive_float, default=0.2, help="Diffusion rate of mice")
    parser.add_argument("-b", "--birth-foxes", type=check_positive_float, default=0.03, help="Birth rate of foxes")
    parser.add_argument("-m", "--death-foxes", type=check_positive_float, default=0.09, help="Rate at which foxes starve")
    parser.add_argument("-l", "--diffusion-foxes", type=check_positive_float, default=0.2, help="Diffusion rate of foxes")
    parser.add_argument("-dt", "--delta-t", type=check_positive_float, default=0.5, help="Time step size")
    parser.add_argument("-t", "--time-step", type=check_positive_int, default=10, help="Number of time steps at which to output files")
    parser.add_argument("-d", "--duration", type=check_positive_int, default=500, help="Time to run the simulation (in timesteps)")
    parser.add_argument("-f", "--landscape-file", type=str, required=True, help="Input landscape file")
    parser.add_argument("-ms", "--mouse-seed", type=int, default=1, help="Random seed for initializing mouse densities")
    parser.add_argument("-fs", "--fox-seed", type=int, default=1, help="Random seed for initializing fox densities")
    return parser.parse_args()

def read_landscape(file_path):
    """Reads the landscape configuration from a file.
    
    Parameters:
    file_path (str): Path to the landscape file.

    Returns:
    tuple: A tuple containing the landscape array, width, and height.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The landscape file '{file_path}' does not exist.")
    
    try:
        with open(file_path, "r") as file:
            width, height = map(int, file.readline().split())
            if width <= 0 or height <= 0:
                raise ValueError("Landscape dimensions must be positive.")
            landscape = np.zeros((height + 2, width + 2), int)
            for row, line in enumerate(file, start=1):
                landscape[row, 1:width+1] = [int(i) for i in line.split()]
    except Exception as e:
        raise RuntimeError(f"Error reading landscape file: {e}")
    
    return landscape, width, height

def initialize_density_array(landscape, seed, height, width):
    """Initialize density array with random values depending on landscape cells.
    
    Parameters:
    landscape (ndarray): The landscape array.
    seed (int): Random seed.
    height (int): Height of the landscape.
    width (int): Width of the landscape.

    Returns:
    ndarray: Initialized density array.
    """
    density_array = landscape.astype(float)
    random.seed(seed)
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            density_array[x, y] = random.uniform(0, 5.0) if landscape[x, y] else 0
    return density_array

def calculate_land_neighbors(landscape, height, width):
    """Calculates the number of land neighbors for each cell in the landscape.
    
    Parameters:
    landscape (ndarray): The landscape array.
    height (int): Height of the landscape.
    width (int): Width of the landscape.

    Returns:
    ndarray: An array of the same shape as landscape with neighbor counts.
    """
    num_neighbors = np.zeros_like(landscape)
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            num_neighbors[x, y] = (landscape[x-1, y] + landscape[x+1, y] +
                                   landscape[x, y-1] + landscape[x, y+1])
    return num_neighbors

def calculate_averages(density, num_land_squares):
    """Calculate the average density over land-only squares.
    
    Parameters:
    density (ndarray): Density array.
    num_land_squares (int): Number of land squares.

    Returns:
    float: Average density.
    """
    return np.sum(density) / num_land_squares if num_land_squares else 0

def write_header():
    """Writes the header for the CSV output file."""
    try:
        with open("averages.csv", "w") as file:
            file.write("Timestep,Time,Mice,Foxes\n")
    except Exception as e:
        raise RuntimeError(f"Error writing header to averages.csv: {e}")

def write_averages(index, delta_t, avg_mice, avg_foxes):
    """Appends average densities to the CSV output file.
    
    Parameters:
    index (int): Current timestep index.
    delta_t (float): Time step size.
    avg_mice (float): Average density of mice.
    avg_foxes (float): Average density of foxes.
    """
    try:
        with open("averages.csv", "a") as file:
            file.write(f"{index},{index * delta_t:.1f},{avg_mice:.17f},{avg_foxes:.17f}\n")
    except Exception as e:
        raise RuntimeError(f"Error writing averages to file: {e}")

def update_densities(landscape, mouse_density, fox_density, new_mouse_density, 
                     new_fox_density, num_neighbors, mouse_rates, fox_rates, 
                     delta_t, height, width):
    """Update densities based on growth, predation, and diffusion rates.
    
    Parameters:
    landscape (ndarray): Landscape array.
    mouse_density (ndarray): Current mouse density array.
    fox_density (ndarray): Current fox density array.
    new_mouse_density (ndarray): Next state of mouse density array.
    new_fox_density (ndarray): Next state of fox density array.
    num_neighbors (ndarray): Number of land neighbors for each cell.
    mouse_rates (tuple): Rates for mouse birth, death, and diffusion.
    fox_rates (tuple): Rates for fox birth, death, and diffusion.
    delta_t (float): Time step size.
    height (int): Height of the landscape.
    width (int): Width of the landscape.
    """
    mouse_birth_rate, mouse_death_rate, mouse_diffusion_rate = mouse_rates
    fox_birth_rate, fox_death_rate, fox_diffusion_rate = fox_rates

    for x in range(1, height + 1):
        for y in range(1, width + 1):
            if landscape[x, y]:
                md = mouse_density[x, y]
                fd = fox_density[x, y]
                mn = num_neighbors[x, y]
                # Updating mouse density
                growth = mouse_birth_rate * md
                predation = mouse_death_rate * md * fd
                diffusion = mouse_diffusion_rate * (
                    (mouse_density[x-1, y] + mouse_density[x+1, y] +
                     mouse_density[x, y-1] + mouse_density[x, y+1]) - mn * md)
                new_mouse_density[x, y] = md + delta_t * (growth - predation + diffusion)
                if new_mouse_density[x, y] < 0:
                    new_mouse_density[x, y] = 0

                # Updating fox density
                fox_growth = fox_birth_rate * md * fd
                starvation = fox_death_rate * fd
                diffusion = fox_diffusion_rate * (
                    (fox_density[x-1, y] + fox_density[x+1, y] +
                     fox_density[x, y-1] + fox_density[x, y+1]) - mn * fd)
                new_fox_density[x, y] = fd + delta_t * (fox_growth - starvation + diffusion)
                if new_fox_density[x, y] < 0:
                    new_fox_density[x, y] = 0

def generate_color_maps(mouse_density, fox_density, landscape, max_mice_density, 
                        max_fox_density, height, width):
    """Generate color maps for visualization of densities.
    
    Parameters:
    mouse_density (ndarray): Density of mice.
    fox_density (ndarray): Density of foxes.
    landscape (ndarray): Landscape array.
    max_mice_density (float): Maximum density of mice.
    max_fox_density (float): Maximum density of foxes.
    height (int): Height of the landscape.
    width (int): Width of the landscape.

    Returns:
    tuple: Color maps for mice and fox densities.
    """
    mouse_color_map = np.zeros((height, width), int)
    fox_color_map = np.zeros((height, width), int)
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            if landscape[x, y]:
                mouse_color_map[x-1, y-1] = (
                    int((mouse_density[x, y] / max_mice_density) * 255)
                    if max_mice_density else 0)
                fox_color_map[x-1, y-1] = (
                    int((fox_density[x, y] / max_fox_density) * 255)
                    if max_fox_density else 0)
    return mouse_color_map, fox_color_map

def write_ppm_file(index, mouse_color_map, fox_color_map, landscape, width, height):
    """Writes the PPM file for visualization at a given timestep.
    
    Parameters:
    index (int): Current timestep index.
    mouse_color_map (ndarray): Color map of mouse densities.
    fox_color_map (ndarray): Color map of fox densities.
    landscape (ndarray): Landscape array.
    width (int): Width of the landscape.
    height (int): Height of the landscape.
    """
    try:
        with open(f"map_{index:04d}.ppm", "w") as file:
            file.write(f"P3\n{width} {height}\n255\n")
            for x in range(height):
                for y in range(width):
                    if landscape[x+1, y+1]:
                        file.write(f"{fox_color_map[x, y]} {mouse_color_map[x, y]} 0\n")
                    else:
                        # Arbitrary color for non-landscaped areas
                        file.write("0 200 255\n")
    except Exception as e:
        raise RuntimeError(f"Error writing to ppm file: {e}")

def print_and_write_averages(index, mouse_density, fox_density, num_land_squares, delta_t):
    """Print and write average densities to the CSV file.
    
    Parameters:
    index (int): Current timestep index.
    mouse_density (ndarray): Density of mice.
    fox_density (ndarray): Density of foxes.
    num_land_squares (int): Number of land squares.
    delta_t (float): Time step size.
    """
    avg_mice = calculate_averages(mouse_density, num_land_squares)
    avg_foxes = calculate_averages(fox_density, num_land_squares)
    print(f"Averages. Timestep: {index} Time (s): {index * delta_t:.1f} Mice: {avg_mice:.17f} Foxes: {avg_foxes:.17f}")
    write_averages(index, delta_t, avg_mice, avg_foxes)

def generate_and_write_maps(index, mouse_density, fox_density, landscape, height, width):
    """Generate and write color maps to PPM files.
    
    Parameters:
    index (int): Current timestep index.
    mouse_density (ndarray): Density of mice.
    fox_density (ndarray): Density of foxes.
    landscape (ndarray): Landscape array.
    height (int): Height of the landscape.
    width (int): Width of the landscape.
    """
    max_mice_density = np.max(mouse_density)
    max_fox_density = np.max(fox_density)
    mouse_color_map, fox_color_map = generate_color_maps(mouse_density, fox_density, 
                                                         landscape, max_mice_density, 
                                                         max_fox_density, height, width)
    write_ppm_file(index, mouse_color_map, fox_color_map, landscape, width, height)

def simulate(mouse_birth_rate, mouse_death_rate, mouse_diffusion_rate, fox_birth_rate, 
             fox_death_rate, fox_diffusion_rate, delta_t, time_step_interval, duration, 
             landscape_file, mouse_seed, fox_seed):
    """Main simulation function that runs the predator-prey model.
    
    Parameters:
    mouse_birth_rate (float): Rate at which mice are born.
    mouse_death_rate (float): Rate at which mice are eaten.
    mouse_diffusion_rate (float): Rate at which mice spread out.
    fox_birth_rate (float): Rate at which foxes are born.
    fox_death_rate (float): Rate at which foxes die.
    fox_diffusion_rate (float): Rate at which foxes spread out.
    delta_t (float): Time step size.
    time_step_interval (int): Interval of steps to output data.
    duration (int): Total duration of the simulation in timesteps.
    landscape_file (str): Path to the landscape configuration file.
    mouse_seed (int): Seed for mouse density initialization.
    fox_seed (int): Seed for fox density initialization.
    """
    print("Predator-prey simulation", getVersion())
    
    landscape, width, height = read_landscape(landscape_file)
    num_land_squares = np.count_nonzero(landscape)
    print(f"Number of land-only squares: {num_land_squares}")

    num_neighbors = calculate_land_neighbors(landscape, height, width)
    
    mouse_density = initialize_density_array(landscape, mouse_seed, height, width)
    fox_density = initialize_density_array(landscape, fox_seed, height, width)

    new_mouse_density = mouse_density.copy()
    new_fox_density = fox_density.copy()

    write_header()
    print_and_write_averages(0, mouse_density, fox_density, num_land_squares, delta_t)

    mouse_rates = (mouse_birth_rate, mouse_death_rate, mouse_diffusion_rate)
    fox_rates = (fox_birth_rate, fox_death_rate, fox_diffusion_rate)
    total_time_steps = int(duration / delta_t)

    for i in range(total_time_steps):
        if i % time_step_interval == 0:
            print_and_write_averages(i, mouse_density, fox_density, num_land_squares, delta_t)
            generate_and_write_maps(i, mouse_density, fox_density, landscape, height, width)

        update_densities(landscape, mouse_density, fox_density, new_mouse_density, 
                         new_fox_density, num_neighbors, mouse_rates, fox_rates, 
                         delta_t, height, width)

        # Swap for next iteration
        mouse_density, new_mouse_density = new_mouse_density, mouse_density
        fox_density, new_fox_density = new_fox_density, fox_density

def sim_comm_line_intf():
    """Handles command-line interface for the simulation."""
    try:
        args = parse_arguments()
        simulate(args.birth_mice, args.death_mice, args.diffusion_mice, args.birth_foxes,
                 args.death_foxes, args.diffusion_foxes, args.delta_t, args.time_step,
                 args.duration, args.landscape_file, args.mouse_seed, args.fox_seed)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sim_comm_line_intf()
