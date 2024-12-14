'''Predator-prey simulation. Foxes and mice.

Version 3.0, last updated in December 2024.
'''
from argparse import ArgumentParser
import numpy as np
import random
import time

def getVersion():
    return 3.0

def initialize_density_array(lscape, seed, h, w):
    density_array = lscape.astype(float).copy()
    random.seed(seed)
    for x in range(1, h + 1):
        for y in range(1, w + 1):
            if seed == 0:
                density_array[x, y] = 0
            else:
                density_array[x, y] = random.uniform(0, 5.0) if lscape[x, y] else 0
    return density_array

def parse_arguments():
    par = ArgumentParser()
    par.add_argument("-r", "--birth-mice", type=float, default=0.1, help="Birth rate of mice")
    par.add_argument("-a", "--death-mice", type=float, default=0.05, help="Rate at which foxes eat mice")
    par.add_argument("-k", "--diffusion-mice", type=float, default=0.2, help="Diffusion rate of mice")
    par.add_argument("-b", "--birth-foxes", type=float, default=0.03, help="Birth rate of foxes")
    par.add_argument("-m", "--death-foxes", type=float, default=0.09, help="Rate at which foxes starve")
    par.add_argument("-l", "--diffusion-foxes", type=float, default=0.2, help="Diffusion rate of foxes")
    par.add_argument("-dt", "--delta-t", type=float, default=0.5, help="Time step size")
    par.add_argument("-t", "--time_step", type=int, default=10, help="Number of time steps at which to output files")
    par.add_argument("-d", "--duration", type=int, default=500, help="Time to run the simulation (in timesteps)")
    par.add_argument("-f", "--landscape-file", type=str, required=True, help="Input landscape file")
    par.add_argument("-ms", "--mouse-seed", type=int, default=1, help="Random seed for initialising mouse densities")
    par.add_argument("-fs", "--fox-seed", type=int, default=1, help="Random seed for initialising fox densities")
    return par.parse_args()

def read_landscape(landscape_file):
    with open(landscape_file, "r") as f:
        width, height = [int(i) for i in f.readline().split(" ")]
        lscape = np.zeros((height + 2, width + 2), int)
        for row, line in enumerate(f.readlines(), start=1):
            lscape[row] = [0] + [int(i) for i in line.split()] + [0]
    return lscape, width, height

def calculate_land_neighbors(lscape, height, width):
    num_neighbors = np.zeros_like(lscape)
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            num_neighbors[x, y] = lscape[x-1, y] + lscape[x+1, y] + lscape[x, y-1] + lscape[x, y+1]
    return num_neighbors

def calculate_averages(density, num_land_squares):
    return np.sum(density) / num_land_squares if num_land_squares else 0

def write_header():
    with open("averages.csv", "w") as f:
        f.write("Timestep,Time,Mice,Foxes\n")

def write_averages(i, delta_t, avg_mice, avg_foxes):
    with open("averages.csv", "a") as f:
        f.write(f"{i},{i * delta_t:.1f},{avg_mice:.17f},{avg_foxes:.17f}\n")

def update_densities(lscape, mouse_density, fox_density, mouse_density_new, fox_density_new, num_neighbors, mouse_rates, fox_rates, delta_t, height, width):
    mouse_birth_rate, mouse_death_rate, mouse_diffusion_rate = mouse_rates
    fox_birth_rate, fox_death_rate, fox_diffusion_rate = fox_rates

    for x in range(1, height + 1):
        for y in range(1, width + 1):
            if lscape[x, y]:
                md = mouse_density[x, y]
                fd = fox_density[x, y]
                mn = num_neighbors[x, y]

                # Update mouse density
                growth = mouse_birth_rate * md
                predation = mouse_death_rate * md * fd
                diffusion = mouse_diffusion_rate * ((mouse_density[x-1, y] + mouse_density[x+1, y] + mouse_density[x, y-1] + mouse_density[x, y+1]) - mn * md)
                mouse_density_new[x, y] = md + delta_t * (growth - predation + diffusion)
                if mouse_density_new[x, y] < 0:
                    mouse_density_new[x, y] = 0

                # Update fox density
                fd = fox_density[x, y]
                fox_growth = fox_birth_rate * md * fd
                starvation = fox_death_rate * fd
                diffusion = fox_diffusion_rate * ((fox_density[x-1, y] + fox_density[x+1, y] + fox_density[x, y-1] + fox_density[x, y+1]) - mn * fd)
                fox_density_new[x, y] = fd + delta_t * (fox_growth - starvation + diffusion)
                if fox_density_new[x, y] < 0:
                    fox_density_new[x, y] = 0

def generate_color_maps(mouse_density, fox_density, lscape, max_mice_density, max_fox_density, height, width):
    mouse_color_map = np.zeros((height, width), int)
    fox_color_map = np.zeros((height, width), int)
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            if lscape[x, y]:
                mouse_color_map[x-1, y-1] = int((mouse_density[x, y] / max_mice_density) * 255) if max_mice_density else 0
                fox_color_map[x-1, y-1] = int((fox_density[x, y] / max_fox_density) * 255) if max_fox_density else 0
    return mouse_color_map, fox_color_map

def write_ppm_file(i, mouse_color_map, fox_color_map, lscape, width, height):
    with open(f"map_{i:04d}.ppm", "w") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for x in range(height):
            for y in range(width):
                if lscape[x+1, y+1]:
                    f.write(f"{fox_color_map[x, y]} {mouse_color_map[x, y]} 0\n")
                else:
                    f.write("0 200 255\n")

def print_and_write_averages(i, mouse_density, fox_density, num_land_squares, delta_t):
    avg_mice = calculate_averages(mouse_density, num_land_squares)
    avg_foxes = calculate_averages(fox_density, num_land_squares)
    print(f"Averages. Timestep: {i} Time (s): {i * delta_t:.1f} Mice: {avg_mice:.17f} Foxes: {avg_foxes:.17f}")
    write_averages(i, delta_t, avg_mice, avg_foxes)

def generate_and_write_maps(i, mouse_density, fox_density, lscape, height, width):
    max_mice_density = np.max(mouse_density)
    max_fox_density = np.max(fox_density)
    mouse_color_map, fox_color_map = generate_color_maps(mouse_density, fox_density, lscape, max_mice_density, max_fox_density, height, width)
    write_ppm_file(i, mouse_color_map, fox_color_map, lscape, width, height)

def sim(mouse_birth_rate, mouse_death_rate, mouse_diffusion_rate, fox_birth_rate, fox_death_rate, fox_diffusion_rate, delta_t, time_step_interval, duration, landscape_file, mouse_seed, fox_seed):
    print("Predator-prey simulation", getVersion())
    
    lscape, width, height = read_landscape(landscape_file)
    num_land_squares = np.count_nonzero(lscape)
    print(f"Number of land-only squares: {num_land_squares}")

    num_neighbors = calculate_land_neighbors(lscape, height, width)
    
    mouse_density = initialize_density_array(lscape, mouse_seed, height, width)
    fox_density = initialize_density_array(lscape, fox_seed, height, width)

    mouse_density_new = mouse_density.copy()
    fox_density_new = fox_density.copy()

    # Initialize output
    write_header()
    print_and_write_averages(0, mouse_density, fox_density, num_land_squares, delta_t)

    mouse_rates = (mouse_birth_rate, mouse_death_rate, mouse_diffusion_rate)
    fox_rates = (fox_birth_rate, fox_death_rate, fox_diffusion_rate)
    total_time_steps = int(duration / delta_t)

    for i in range(total_time_steps):
        if i % time_step_interval == 0:
            print_and_write_averages(i, mouse_density, fox_density, num_land_squares, delta_t)
            generate_and_write_maps(i, mouse_density, fox_density, lscape, height, width)

        update_densities(lscape, mouse_density, fox_density, mouse_density_new, fox_density_new, num_neighbors, mouse_rates, fox_rates, delta_t, height, width)

        # Swap arrays for next iteration
        mouse_density, mouse_density_new = mouse_density_new, mouse_density
        fox_density, fox_density_new = fox_density_new, fox_density

def simCommLineIntf():
    args = parse_arguments()
    sim(args.birth_mice, args.death_mice, args.diffusion_mice, args.birth_foxes,
        args.death_foxes, args.diffusion_foxes, args.delta_t, args.time_step,
        args.duration, args.landscape_file, args.mouse_seed, args.fox_seed)

if __name__ == "__main__":
    simCommLineIntf()
