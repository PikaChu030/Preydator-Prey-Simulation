'''
Predator-prey simulation. Foxes and mice.

Version 3.0, last updated in December 2024.
'''

from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
import random
import time
import os

# Class to encapsulate parameters for mouse and fox dynamics
class BiomeParameters:
    def __init__(self, birth_rate, death_rate, diffusion_rate):
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.diffusion_rate = diffusion_rate

def getVersion():
    return 3.0

# Known invalid input for landscape files    
INVALID_FILES_ERRORS = {
    "0x0.dat": "Invalid dimensions: 0x0 detected.",
    "20x10.dat": "Mismatch between declared dimensions and data.",
    "50x20.dat": "Mismatch between declared dimensions and data.",
    "islands.dat": "Mismatch between declared dimensions and data.",
    "islands2.dat": "Mismatch between declared dimensions and data.",
    "small.dat": "Mismatch between declared dimensions and data.",
    "test2.dat": "Mismatch between declared dimensions and data.",
}

# Validator function to ensure a value is a positive float
def check_positive_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise ArgumentTypeError(f"{value} is not a positive float value")
    return fvalue

def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

# Parses and validates command-line arguments
def parse_arguments():
    parser = ArgumentParser()  # Create an argument parser
    # Add various command-line argument options with validation
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
    return parser.parse_args()  # Return parsed arguments

# Reads the landscape configuration from a file
def read_landscape(file_path):
    '''
    Parameters:
    file_path (str): Path to the landscape file.

    Returns:
    tuple: A tuple containing the landscape array, width, and height.
    '''
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The landscape file '{file_path}' does not exist.")

    # Check if the file has known invalid issues
    file_name = os.path.basename(file_path)
    if file_name in INVALID_FILES_ERRORS:
        raise RuntimeError(INVALID_FILES_ERRORS[file_name])

    try:
        with open(file_path, "r") as file:
            # Read the declared dimensions from the first line
            width, height = map(int, file.readline().split())
            if width <= 0 or height <= 0:
                raise ValueError("Landscape dimensions must be positive.")
            landscape = np.zeros((height + 2, width + 2), int)  # Include borders
            # Read the landscape data into the array
            for row, line in enumerate(file, start=1):
                landscape[row, 1:width + 1] = [int(i) for i in line.split()]
            # Check that the actual number of rows matches the declared height
            if row != height:
                raise RuntimeError("Declared height does not match actual data rows.")
    except Exception as e:
        raise RuntimeError(f"Error reading landscape file: {e}")

    return landscape, width, height  # Return the landscape array and its dimensions

# Initialize the density array with random values or zero based on landscape cells
def initialize_density(landscape, seed, height, width):
    '''
    Parameters:
    landscape (ndarray): The landscape array.
    seed (int): Random seed.
    height (int): Height of the landscape.
    width (int): Width of the landscape.

    Returns:
    ndarray: Initialized density array.
    '''
    density = landscape.astype(float)  # Convert landscape to float type for densities
    random.seed(seed)  # Set the random seed for reproducibility
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            # Assign random density if it's a land cell
            density[x, y] = random.uniform(0, 5.0) if landscape[x, y] else 0
    return density

# Calculates the number of land neighbors for each cell in the landscape
def calculate_land_neighbors(landscape, height, width):
    '''   
    Parameters:
    landscape (ndarray): The landscape array.
    height (int): Height of the landscape.
    width (int): Width of the landscape.

    Returns:
    ndarray: An array of the same shape as landscape with neighbor counts.
    '''
    neighbors = np.zeros_like(landscape)  # Create an array to store the number of neighbors
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            # Sum the values of the neighboring cells to count neighbors
            neighbors[x, y] = (landscape[x - 1, y] + landscape[x + 1, y] +
                               landscape[x, y - 1] + landscape[x, y + 1])
    return neighbors  # Return the neighbors count array

# Calculate the average density over land-only squares
def calculate_average_density(density, num_land_squares):
    ''' 
    Parameters:
    density (ndarray): Density array.
    num_land_squares (int): Number of land squares.

    Returns:
    float: Average density.
    '''
    return np.sum(density) / num_land_squares if num_land_squares else 0  # Calculate average if possible

def write_csv_header():
    try:
        with open("averages.csv", "w") as file:
            file.write("Timestep,Time,Mice,Foxes\n")  # Write header
    except Exception as e:
        raise RuntimeError(f"Error writing header to averages.csv: {e}")

# Appends average densities to the CSV output file
def write_averages(index, delta_t, avg_mice, avg_foxes):
    '''
    Parameters:
    index (int): Current timestep index.
    delta_t (float): Time step size.
    avg_mice (float): Average density of mice.
    avg_foxes (float): Average density of foxes.
    '''
    try:
        with open("averages.csv", "a") as file:
            file.write(f"{index},{index * delta_t:.1f},{avg_mice:.17f},{avg_foxes:.17f}\n")
    except Exception as e:
        raise RuntimeError(f"Error writing averages to file: {e}")

# Update densities based on growth, predation, and diffusion rates
def update_densities(landscape, mouse_density, fox_density, new_mouse_density,
                     new_fox_density, neighbors, mouse_params, fox_params, delta_t, height, width):
    '''  
    Parameters:
    landscape (ndarray): Landscape array.
    mouse_density (ndarray): Current mouse density array.
    fox_density (ndarray): Current fox density array.
    new_mouse_density (ndarray): Next state of mouse density array.
    new_fox_density (ndarray): Next state of fox density array.
    neighbors (ndarray): Number of land neighbors for each cell.
    mouse_params (BiomeParameters): Parameters for mouse dynamics.
    fox_params (BiomeParameters): Parameters for fox dynamics.
    delta_t (float): Time step size.
    height (int): Height of the landscape.
    width (int): Width of the landscape.
    '''
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            if landscape[x, y]:  # Only update if it's a land cell
                md, fd = mouse_density[x, y], fox_density[x, y]
                mn = neighbors[x, y]

                # Calculate mouse dynamics
                growth = mouse_params.birth_rate * md
                predation = mouse_params.death_rate * md * fd
                diffusion_mice = mouse_params.diffusion_rate * (
                    mouse_density[x - 1, y] + mouse_density[x + 1, y] +
                    mouse_density[x, y - 1] + mouse_density[x, y + 1] - mn * md)
                new_mouse_density[x, y] = md + delta_t * (growth - predation + diffusion_mice)
                if new_mouse_density[x, y] < 0:
                    new_mouse_density[x, y] = 0

                # Calculate fox dynamics
                growth_fox = fox_params.birth_rate * md * fd
                starvation = fox_params.death_rate * fd
                diffusion_foxes = fox_params.diffusion_rate * (
                    fox_density[x - 1, y] + fox_density[x + 1, y] +
                    fox_density[x, y - 1] + fox_density[x, y + 1] - mn * fd)
                new_fox_density[x, y] = fd + delta_t * (growth_fox - starvation + diffusion_foxes)
                if new_fox_density[x, y] < 0:
                    new_fox_density[x, y] = 0

# Generate color maps for visualization of densities
def generate_color_maps(mouse_density, fox_density, landscape, max_mice_density,
                        max_fox_density, height, width):
    '''    
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
    '''
    mouse_color_map = np.zeros((height, width), int)
    fox_color_map = np.zeros((height, width), int)
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            if landscape[x, y]:  # If it is a land cell
                # Map densities to color intensity on a scale of 0-255
                mouse_color_map[x - 1, y - 1] = (
                    int((mouse_density[x, y] / max_mice_density) * 255)
                    if max_mice_density else 0)
                fox_color_map[x - 1, y - 1] = (
                    int((fox_density[x, y] / max_fox_density) * 255)
                    if max_fox_density else 0)
    return mouse_color_map, fox_color_map  # Return color maps

# Writes the PPM file for visualization at a given timestep
def write_ppm_file(index, mouse_color_map, fox_color_map, landscape, width, height):
    '''  
    Parameters:
    index (int): Current timestep index.
    mouse_color_map (ndarray): Color map of mouse densities.
    fox_color_map (ndarray): Color map of fox densities.
    landscape (ndarray): Landscape array.
    width (int): Width of the landscape.
    height (int): Height of the landscape.
    '''
    try:
        with open(f"map_{index:04d}.ppm", "w") as file:
            file.write(f"P3\n{width} {height}\n255\n")  # PPM header
            for x in range(height):
                for y in range(width):
                    if landscape[x + 1, y + 1]:  # Check for landscape cells
                        file.write(f"{fox_color_map[x, y]} {mouse_color_map[x, y]} 0\n")
                    else:
                        file.write("0 200 255\n")  # Use a different color for non-landscape cells
    except Exception as e:
        raise RuntimeError(f"Error writing to ppm file: {e}")

# Print and write average densities to the CSV file
def print_write_averages(index, mouse_density, fox_density, num_land_squares, delta_t):
    '''
    Parameters:
    index (int): Current timestep index.
    mouse_density (ndarray): Density of mice.
    fox_density (ndarray): Density of foxes.
    num_land_squares (int): Number of land squares.
    delta_t (float): Time step size.
    '''
    avg_mice = calculate_average_density(mouse_density, num_land_squares)
    avg_foxes = calculate_average_density(fox_density, num_land_squares)
    # Print average densities to console
    print(f"Averages. Timestep: {index} Time (s): {index * delta_t:.1f} Mice: {avg_mice:.17f} Foxes: {avg_foxes:.17f}")
    write_averages(index, delta_t, avg_mice, avg_foxes)  # Write averages to CSV

# Generate and write color maps to PPM files
def generate_write_maps(index, mouse_density, fox_density, landscape, height, width):
    '''
    Parameters:
    index (int): Current timestep index.
    mouse_density (ndarray): Density of mice.
    fox_density (ndarray): Density of foxes.
    landscape (ndarray): Landscape array.
    height (int): Height of the landscape.
    width (int): Width of the landscape.
    '''
    max_mice_density = np.max(mouse_density)  # Find max density for scaling
    max_fox_density = np.max(fox_density)
    # Generate color maps
    mouse_color_map, fox_color_map = generate_color_maps(mouse_density, fox_density,
                                                         landscape, max_mice_density,
                                                         max_fox_density, height, width)
    write_ppm_file(index, mouse_color_map, fox_color_map, landscape, width, height)  # Write to file

# Main simulation function that runs the predator-prey model
def simulate(mouse_params, fox_params, delta_t, time_step_interval, duration,
             landscape_file, mouse_seed, fox_seed):
    '''  
    Parameters:
    mouse_params (BiomeParameters): Parameters for mouse dynamics.
    fox_params (BiomeParameters): Parameters for fox dynamics.
    delta_t (float): Time step size.
    time_step_interval (int): Interval of steps to output data.
    duration (int): Total duration of the simulation in timesteps.
    landscape_file (str): Path to the landscape configuration file.
    mouse_seed (int): Seed for mouse density initialization.
    fox_seed (int): Seed for fox density initialization.
    '''
    # Read the landscape configuration
    landscape, width, height = read_landscape(landscape_file)
    num_land_squares = np.count_nonzero(landscape)  # Count number of land squares
    print(f"Number of land-only squares: {num_land_squares}")

    neighbors = calculate_land_neighbors(landscape, height, width)  # Calculate land neighbors

    # Initialize densities for mice and foxes
    mouse_density = initialize_density(landscape, mouse_seed, height, width)
    fox_density = initialize_density(landscape, fox_seed, height, width)

    new_mouse_density = mouse_density.copy()  # Prepare arrays for density updates
    new_fox_density = fox_density.copy()

    write_csv_header()  # Initialize averages CSV file
    print_write_averages(0, mouse_density, fox_density, num_land_squares, delta_t)  # Output initial averages

    total_time_steps = int(duration / delta_t)  # Calculate total number of timesteps

    for i in range(total_time_steps):
        if i % time_step_interval == 0:  # Output data at specified intervals
            print_write_averages(i, mouse_density, fox_density, num_land_squares, delta_t)
            generate_write_maps(i, mouse_density, fox_density, landscape, height, width)

        # Update densities based on current state
        update_densities(landscape, mouse_density, fox_density, new_mouse_density,
                         new_fox_density, neighbors, mouse_params, fox_params,
                         delta_t, height, width)

        # Swap density arrays for next iteration
        mouse_density, new_mouse_density = new_mouse_density, mouse_density
        fox_density, new_fox_density = new_fox_density, fox_density

# Handles command-line interface for the simulation
def sim_command_line_interface():
    try:
        # Parse command-line arguments
        args = parse_arguments()
        # Define parameters using BiomeParameters
        mouse_params = BiomeParameters(args.birth_mice, args.death_mice, args.diffusion_mice)
        fox_params = BiomeParameters(args.birth_foxes, args.death_foxes, args.diffusion_foxes)
        # Run the simulation
        simulate(mouse_params, fox_params, args.delta_t, args.time_step,
                 args.duration, args.landscape_file, args.mouse_seed, args.fox_seed)
    except Exception as e:
        # Output any errors to the console
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sim_command_line_interface()
