'''Predator-prey simulation. Foxes and mice.

Version 3.0, last updated in September 2023.
'''
from argparse import ArgumentParser
import numpy as np
import random
import time

def getVersion():
    return 3.0

def simCommLineIntf():
    par=ArgumentParser()
    par.add_argument("-r","--birth-mice",type=float,default=0.1,help="Birth rate of mice")
    par.add_argument("-a","--death-mice",type=float,default=0.05,help="Rate at which foxes eat mice")
    par.add_argument("-k","--diffusion-mice",type=float,default=0.2,help="Diffusion rate of mice")
    par.add_argument("-b","--birth-foxes",type=float,default=0.03,help="Birth rate of foxes")
    par.add_argument("-m","--death-foxes",type=float,default=0.09,help="Rate at which foxes starve")
    par.add_argument("-l","--diffusion-foxes",type=float,default=0.2,help="Diffusion rate of foxes")
    par.add_argument("-dt","--delta-t",type=float,default=0.5,help="Time step size")
    par.add_argument("-t","--time_step",type=int,default=10,help="Number of time steps at which to output files")
    par.add_argument("-d","--duration",type=int,default=500,help="Time to run the simulation (in timesteps)")
    par.add_argument("-f","--landscape-file",type=str,required=True,
                        help="Input landscape file")
    par.add_argument("-ms","--mouse-seed",type=int,default=1,help="Random seed for initialising mouse densities")
    par.add_argument("-fs","--fox-seed",type=int,default=1,help="Random seed for initialising fox densities")
    args=par.parse_args()
    sim(args.birth_mice,args.death_mice,args.diffusion_mice,args.birth_foxes,args.death_foxes,args.diffusion_foxes,args.delta_t,args.time_step,args.duration,args.landscape_file,args.mouse_seed,args.fox_seed)

def sim(r,a,k,b,m,l,dt,t,d,lfile,mseed,fseed):
    print("Predator-prey simulation",getVersion())
    with open(lfile,"r") as f:
        w,h=[int(i) for i in f.readline().split(" ")]
        print("Width: {} Height: {}".format(w,h))
        wh=w+2 # Width including halo
        hh=h+2 # Height including halo
        lscape=np.zeros((hh,wh),int)
        row=1
        for line in f.readlines():
            values=line.split(" ")
            # Read landscape into array,padding with halo values.
            lscape[row]=[0]+[int(i) for i in values]+[0]
            row += 1
    nlands=np.count_nonzero(lscape)
    print("Number of land-only squares: {}".format(nlands))
    # Pre-calculate number of land neighbours of each land square.
    neibs=np.zeros((hh,wh),int)
    for x in range(1,h+1):
        for y in range(1,w+1):
            neibs[x,y]=lscape[x-1,y] \
                + lscape[x+1,y] \
                + lscape[x,y-1] \
                + lscape[x,y+1]
    ms=lscape.astype(float).copy()
    fs=lscape.astype(float).copy()
    random.seed(mseed)
    for x in range(1,h+1):
        for y in range(1,w+1):
            if mseed==0:
                ms[x,y]=0
            else:
                if lscape[x,y]:
                    ms[x,y]=random.uniform(0,5.0)
                else:
                    ms[x,y]=0
    random.seed(fseed)
    for x in range(1,h+1):
        for y in range(1,w+1):
            if fseed==0:
                fs[x,y]=0
            else:
                if lscape[x,y]:
                    fs[x,y]=random.uniform(0,5.0)
                else:
                    fs[x,y]=0
    # Create copies of initial maps and arrays for PPM file maps.
    # Reuse these so we don't need to create new arrays going
    # round the simulation loop.
    ms_nu=ms.copy()
    fs_nu=fs.copy()
    mcols=np.zeros((h,w),int)
    fcols=np.zeros((h,w),int)
    if nlands != 0:
        am=np.sum(ms)/nlands
        af=np.sum(fs)/nlands
    else:
        am=0
        af=0
    print("Averages. Timestep: {} Time (s): {:.1f} Mice: {:.17f} Foxes: {:.17f}".format(0,0,am,af))
    with open("averages.csv","w") as f:
        hdr="Timestep,Time,Mice,Foxes\n"
        f.write(hdr)
    tot_ts = int(d / dt)
    for i in range(0,tot_ts):
        if not i%t:
            mm=np.max(ms)
            mf=np.max(fs)
            if nlands != 0:
                am=np.sum(ms)/nlands
                af=np.sum(fs)/nlands
            else:
                am=0
                af=0
            print("Averages. Timestep: {} Time (s): {:.1f} Mice: {:.17f} Foxes: {:.17f}".format(i,i*dt,am,af))
            with open("averages.csv","a") as f:
                f.write("{},{:.1f},{:.17f},{:.17f}\n".format(i,i*dt,am,af))
            for x in range(1,h+1):
                for y in range(1,w+1):
                    if lscape[x,y]:
                        if mm != 0:
                            mcol=(ms[x,y]/mm)*255
                        else:
                            mcol = 0
                        if mf != 0:
                            fcol=(fs[x,y]/mf)*255
                        else:
                            fcol = 0
                        mcols[x-1,y-1]=mcol
                        fcols[x-1,y-1]=fcol
            with open("map_{:04d}.ppm".format(i),"w") as f:
                hdr="P3\n{} {}\n{}\n".format(w,h,255)
                f.write(hdr)
                for x in range(0,h):
                    for y in range(0,w):
                        if lscape[x+1,y+1]:
                            f.write("{} {} {}\n".format(fcols[x,y],mcols[x,y],0))
                        else:
                            f.write("{} {} {}\n".format(0,200,255))
        for x in range(1,h+1):
            for y in range(1,w+1):
                if lscape[x,y]:
                    ms_nu[x,y]=ms[x,y]+dt*((r*ms[x,y])-(a*ms[x,y]*fs[x,y])+k*((ms[x-1,y]+ms[x+1,y]+ms[x,y-1]+ms[x,y+1])-(neibs[x,y]*ms[x,y])))
                    if ms_nu[x,y]<0:
                        ms_nu[x,y]=0
                    fs_nu[x,y]=fs[x,y]+dt*((b*ms[x,y]*fs[x,y])-(m*fs[x,y])+l*((fs[x-1,y]+fs[x+1,y]+fs[x,y-1]+fs[x,y+1])-(neibs[x,y]*fs[x,y])))
                    if fs_nu[x,y]<0:
                        fs_nu[x,y]=0
        # Swap arrays for next iteration.
        tmp=ms
        ms=ms_nu
        ms_nu=tmp
        tmp=fs
        fs=fs_nu
        fs_nu=tmp

if __name__ == "__main__":
    simCommLineIntf()
