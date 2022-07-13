import rawpy
from astropy.io import fits
import numpy as np
import os, imageio
import matplotlib.pyplot as plt

# All of the plt. commented out lines are to plot the resultsat each step of the process. 
#   Such images are shown in the documentation but the code is left here for convenience

def InitialiseClouds():
    files = os.listdir("/home/s1901554/Downloads")
    files = [f for f in files if "Cloud" in f]
    red_stack, green_stack, blue_stack = [], [], []
    for f in files:
        print(f)        
        raw = rawpy.imread("/home/s1901554/Downloads/"+f)
        rgb = raw.postprocess()
        red_stack.append(rgb[:,:,0])
        green_stack.append(rgb[:,:,1])
        blue_stack.append(rgb[:,:,2])
    red_stack, green_stack, blue_stack = np.array(red_stack), np.array(green_stack), np.array(blue_stack) 
    mean_r = np.mean(red_stack, axis=0)
    mean_g = np.mean(green_stack, axis=0)
    mean_b = np.mean(blue_stack, axis=0)
    mean_r_norm = mean_r / np.max(mean_r)
    mean_g_norm = mean_g / np.max(mean_g)
    mean_b_norm = mean_b / np.max(mean_b)
    np.savetxt("red_master_flat.txt", mean_r_norm, fmt='%.8f')
    np.savetxt("green_master_flat.txt", mean_g_norm, fmt='%.8f')
    np.savetxt("blue_master_flat.txt", mean_b_norm, fmt='%.8f')
    
# Commented out since I have already produced the master flat text files for a set of cloudy images. See docs for info.
# InitialiseClouds()
mean_r_norm = np.loadtxt("/home/s1901554/Documents/SpaceTrafficManagement/red_master_flat.txt").astype(float)
mean_g_norm = np.loadtxt("/home/s1901554/Documents/SpaceTrafficManagement/green_master_flat.txt").astype(float)
mean_b_norm = np.loadtxt("/home/s1901554/Documents/SpaceTrafficManagement/blue_master_flat.txt").astype(float)
# plt.imshow(mean_r_norm)
# plt.gca().set_axis_off()
# plt.savefig("R Channel Normalised Master Flat")
# plt.imshow(mean_g_norm)
# plt.gca().set_axis_off()
# plt.savefig("G Channel Normalised Master Flat")
# plt.imshow(mean_b_norm)
# plt.gca().set_axis_off()
# plt.savefig("B Channel Normalised Master Flat")

# Gets list of NEF files in a given directory
files = os.listdir("/home/s1901554/Documents/SpaceTrafficManagement/NEFs")
for f in files:
    # Loads each NEF file
    raw = rawpy.imread("/home/s1901554/Documents/SpaceTrafficManagement/NEFs/"+f)
    # Processes file
    rgb = raw.postprocess()
    # imageio.imsave('ExampleNEF.png', rgb)

    # Extracts RGB separately
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    # plt.imshow(r)
    # plt.gca().set_axis_off()
    # plt.savefig("R Channel Raw", bbox_inches='tight')
    # plt.imshow(g)
    # plt.gca().set_axis_off()
    # plt.savefig("G Channel Raw", bbox_inches='tight')
    # plt.imshow(b)
    # plt.gca().set_axis_off()
    # plt.savefig("B Channel Raw", bbox_inches='tight')
    # Divides by colour-specific master normalised flat field
    r = np.divide(r, mean_r_norm)
    g = np.divide(g, mean_g_norm)
    b = np.divide(b, mean_b_norm)

    # Recombine to one RGB image array
    rgb = np.dstack((r,g,b))

    # Save flatfielded imag as png (could be whatever format you wanted - or could integrate this 
    #     into image processesing part of pipeline on Cuillin before the NEFs were originally converted to greyscale or pngs etc.)
    imageio.imsave('ExampleNEF_Corrected.png', rgb) # this line can be replaced with however one wants to save the flatfielded image
    
    # plt.imshow(r)
    # plt.gca().set_axis_off()
    # plt.savefig("R Channel Corrected", bbox_inches='tight')
    # plt.imshow(g)
    # plt.gca().set_axis_off()
    # plt.savefig("G Channel Corrected", bbox_inches='tight')
    # plt.imshow(b)
    # plt.gca().set_axis_off()
    # plt.savefig("B Channel Corrected", bbox_inches='tight')


