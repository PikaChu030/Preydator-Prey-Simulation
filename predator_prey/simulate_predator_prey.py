from argparse import ArgumentParser
import numpy as np
import random
import time

def getVersion():
    return 3.0

def initialize_density_array(lscape, seed, h, w):
    """
    Initialize a density array based on the given landscape and seed.

    Parameters:
    - lscape: 2D numpy array representing the landscape.
    - seed: Random seed for initialization.
    - h: Height of the landscape.
    - w: Width of the landscape.

    Returns:
    - density_array: Initialized density array.
    """
    density_array = lscape.astype(float).copy()
    random.seed(seed)
    for x in range(1, h+1):
        for y in range(1, w+1):
            if seed == 0:
                density_array[x, y] = 0
            else:
                if lscape[x, y]:
                    density_array[x, y] = random.uniform(0, 5.0)
                else:
                    density_array[x, y] = 0
    return density_array

def simCommLineIntf():
    par=ArgumentParser()
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
    args = par.parse_args()
    sim(args.birth_mice, args.death_mice, args.diffusion_mice, args.birth_foxes,
        args.death_foxes, args.diffusion_foxes, args.delta_t, args.time_step,
        args.duration, args.landscape_file, args.mouse_seed, args.fox_seed)

def sim(mouse_birth_rate, mouse_death_rate, mouse_diffusion_rate, fox_birth_rate, fox_death_rate, fox_diffusion_rate, delta_t, time_step_interval, duration, landscape_file, mouse_seed, fox_seed):
    print("Predator-prey simulation", getVersion())
    with open(landscape_file, "r") as f:
        width, height = [int(i) for i in f.readline().split(" ")]
        print("Width: {} Height: {}".format(width, height))
        width_with_halo = width + 2  # Width including halo
        height_with_halo = height + 2  # Height including halo
        lscape = np.zeros((height_with_halo, width_with_halo), int)
        row = 1
        for line in f.readlines():
            values = line.split(" ")
            # Read landscape into array, padding with halo values.
            lscape[row] = [0] + [int(i) for i in values] + [0]
            row += 1
    num_land_squares = np.count_nonzero(lscape)
    print("Number of land-only squares: {}".format(num_land_squares))
    # Pre-calculate number of land neighbors of each land square.
    num_neighbors = np.zeros((height_with_halo, width_with_halo), int)
    for x in range(1, height+1):
        for y in range(1, width+1):
            num_neighbors[x, y] = lscape[x-1, y] + lscape[x+1, y] + lscape[x, y-1] + lscape[x, y+1]

    mouse_density = initialize_density_array(lscape, mouse_seed, height, width)
    fox_density = initialize_density_array(lscape, fox_seed, height, width)

    # Create copies of initial maps and arrays for PPM file maps.
    # Reuse these so we don't need to create new arrays going
    # round the simulation loop.
    mouse_density_new = mouse_density.copy()
    fox_density_new = fox_density.copy()
    mouse_color_map = np.zeros((height, width), int)
    fox_color_map = np.zeros((height, width), int)
    if num_land_squares != 0:
        avg_mice = np.sum(mouse_density) / num_land_squares
        avg_foxes = np.sum(fox_density) / num_land_squares
    else:
        avg_mice = 0
        avg_foxes = 0
    print("Averages. Timestep: {} Time (s): {:.1f} Mice: {:.17f} Foxes: {:.17f}".format(0, 0, avg_mice, avg_foxes))
    with open("averages.csv", "w") as f:
        header = "Timestep,Time,Mice,Foxes\n"
        f.write(header)
    total_time_steps = int(duration / delta_t)
    for i in range(total_time_steps):
        if not i % time_step_interval:
            max_mice_density = np.max(mouse_density)
            max_fox_density = np.max(fox_density)
            if num_land_squares != 0:
                avg_mice = np.sum(mouse_density) / num_land_squares
                avg_foxes = np.sum(fox_density) / num_land_squares
            else:
                avg_mice = 0
                avg_foxes = 0
            print("Averages. Timestep: {} Time (s): {:.1f} Mice: {:.17f} Foxes: {:.17f}".format(i, i*delta_t, avg_mice, avg_foxes))
            with open("averages.csv", "a") as f:
                f.write("{},{:.1f},{:.17f},{:.17f}\n".format(i, i*delta_t, avg_mice, avg_foxes))
            for x in range(1, height+1):
                for y in range(1, width+1):
                    if lscape[x, y]:
                        if max_mice_density != 0:
                            mouse_color = (mouse_density[x, y] / max_mice_density) * 255
                        else:
                            mouse_color = 0
                        if max_fox_density != 0:
                            fox_color = (fox_density[x, y] / max_fox_density) * 255
                        else:
                            fox_color = 0
                        mouse_color_map[x-1, y-1] = mouse_color
                        fox_color_map[x-1, y-1] = fox_color
            with open("map_{:04d}.ppm".format(i), "w") as f:
                header = "P3\n{} {}\n{}\n".format(width, height, 255)
                f.write(header)
                for x in range(height):
                    for y in range(width):
                        if lscape[x+1, y+1]:
                            f.write("{} {} {}\n".format(fox_color_map[x, y], mouse_color_map[x, y], 0))
                        else:
                            f.write("{} {} {}\n".format(0, 200, 255))
        for x in range(1, height+1):
            for y in range(1, width+1):
                if lscape[x, y]:
                    mouse_density_new[x, y] = mouse_density[x, y] + delta_t * (
                        (mouse_birth_rate * mouse_density[x, y]) -
                        (mouse_death_rate * mouse_density[x, y] * fox_density[x, y]) +
                        mouse_diffusion_rate * (
                            (mouse_density[x-1, y] + mouse_density[x+1, y] + mouse_density[x, y-1] + mouse_density[x, y+1]) -
                            (num_neighbors[x, y] * mouse_density[x, y])
                        )
                    )
                    if mouse_density_new[x, y] < 0:
                        mouse_density_new[x, y] = 0
                    fox_density_new[x, y] = fox_density[x, y] + delta_t * (
                        (fox_birth_rate * mouse_density[x, y] * fox_density[x, y]) -
                        (fox_death_rate * fox_density[x, y]) +
                        fox_diffusion_rate * (
                            (fox_density[x-1, y] + fox_density[x+1, y] + fox_density[x, y-1] + fox_density[x, y+1]) -
                            (num_neighbors[x, y] * fox_density[x, y])
                        )
                    )
                    if fox_density_new[x, y] < 0:
                        fox_density_new[x, y] = 0
        # Swap arrays for next iteration.
        tmp = mouse_density
        mouse_density = mouse_density_new
        mouse_density_new = tmp
        tmp = fox_density
        fox_density = fox_density_new
        fox_density_new = tmp

if __name__ == "__main__":
    simCommLineIntf()
