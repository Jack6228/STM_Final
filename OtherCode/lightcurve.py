from audioop import maxpp
from multiprocessing.sharedctypes import RawValue
import rawpy, os
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from scipy import ndimage
from datetime import datetime, timedelta
from math import atan, degrees, sqrt
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.optimize import curve_fit
from astropy.io import fits

pd.options.mode.chained_assignment = None
exposure_time = 5

def legend_without_duplicate_labels(ax):
    """
    Removes duplicate legend entries from ax
    :param ax: axis to remove entries from
    :return no_objects: number of satellites in the image - calculated from number of legend entries
    """
    # Gets legend data, compiles into a unique list and resets as the legend.
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    return len(unique)

output_data = pd.read_csv("output_data_2022-05-28.csv")
for i in range(len(output_data)):
    output_data['Filename'].iloc[i] = output_data['Filename'].iloc[i][:-13] + '.NEF'

output_data["start"], output_data["end"] = None, None
for i in range(len(output_data)):
    output_data["start"].iloc[i] = datetime.strptime(output_data['Filename'].iloc[i][4:21],'%Y-%m-%d_%H%M%S')
    output_data["end"].iloc[i] = output_data["start"].iloc[i] + timedelta(seconds=exposure_time)

streaks = pd.read_csv("2022-05-28/streaks_data.txt", header=None)
streaks.columns = ['Filename','RA1','Dec1','x1','y1','RA2','Dec2','x2','y2']

uniq = np.unique(output_data['NORAD_CAT_ID'])

def GetWidthHeight(exposure_time, mini, times, ws, hs, f, grad, y_int, r, r_av):
    widths = []
    heights = []
    width_x = 20
    for i in range(len(r)):
        y = i
        x = (y - y_int) / grad
        this_arr = r[y][int(x-width_x):int(x+width_x)]
                # plt.imshow(r)
                # plt.plot([x-width_x,x+width_x],[i,i])
        peaks, _ = find_peaks(this_arr)#, height=r_av)
        results_full = peak_widths(this_arr, peaks, rel_height=1)
        prominences = peak_prominences(this_arr, peaks)[0]
        if len(peaks) > 0:
            max_peak_ind = np.argmax(prominences)
            max_peak = peaks[max_peak_ind]
            width = results_full[0][max_peak_ind]
            height = prominences[max_peak_ind]
                    # print("x =",max_peak)
                    # print("peak intensity =",this_arr[max_peak])
                    # print("height =",height)
                    # print(width)
            widths.append(width)
            heights.append(height-r_av)
                    # plt.plot(this_arr)
                    # plt.plot(peaks, this_arr[peaks], "x")
                    # plt.vlines(x=peaks, ymin=this_arr[peaks] - prominences, ymax=this_arr[peaks])
                    # plt.hlines(*results_full[1:], color="C3")
                    # plt.show()

    time = np.linspace(mini['start_sec'].iloc[f], mini['start_sec'].iloc[f] + exposure_time,len(heights))
    times.extend(time)
    ws.extend(widths)
    hs.extend(heights)
    return ws, hs, times

for i in uniq:
    mini = output_data[output_data['NORAD_CAT_ID'] == i].reset_index(drop=True)
    if len(mini) > 1:
        min_start = np.min(mini["start"])
        mini['start_sec'] = None
        for j in range(len(mini)):
            mini['start_sec'].iloc[j] = (mini["start"].iloc[j] - min_start).total_seconds()

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)

        for f in range(len(mini)):
            filename = mini['Filename'].iloc[f]
            mini_streaks = streaks[streaks['Filename']==filename]
            try:
                raw = rawpy.imread("NEFs/"+filename)
                       
                x1, x2 = min(int(mini_streaks['x1'].item()), int(mini_streaks['x2'].item())), max(int(mini_streaks['x1'].item()), int(mini_streaks['x2'].item()))
                y1, y2 = min(int(mini_streaks['y1'].item()), int(mini_streaks['y2'].item())), max(int(mini_streaks['y1'].item()), int(mini_streaks['y2'].item()))
                grad = (y2 - y1) / (x2 - x1)
                y_int = y2 - (grad * x2)

                angle = degrees( atan( (y2-y1) / (x2-x1) ) )
                # Convert to standard RGB pixels [0:255]
                rgb = raw.postprocess()
                height, width = len(rgb), len(rgb[0])
                r = rgb[:,:,0]
                r_av = np.average(np.ravel(r))
                g = rgb[:,:,1]
                g_av = np.average(np.ravel(g))
                b = rgb[:,:,2]
                b_av = np.average(np.ravel(b))
                sum = np.sum((r,g,b), axis=0)
                sum_av = np.average(np.ravel(b))
                ave = np.mean((r, b, g), axis=0)
                ave_av = np.average(np.ravel(b))

                offset = 0
                r = r[y1-offset:y2+offset,x1-offset:x2+offset]
                g = g[y1-offset:y2+offset,x1-offset:x2+offset]
                b = b[y1-offset:y2+offset,x1-offset:x2+offset]
                sum = sum[y1-offset:y2+offset,x1-offset:x2+offset]
                ave = ave[y1-offset:y2+offset,x1-offset:x2+offset]
                x1, x2 = offset, (x2-x1)-offset
                y1, y2 = offset, (y2-y1)-offset
                grad = (y2 - y1) / (x2 - x1)
                y_int = y2 - (grad * x2)
                mid = len(r)/2

                cols = ['r', 'g', 'b', 'magenta', 'orange']
                imgs = [r, g, b, sum, ave]
                img_avs = [r_av, g_av, b_av, sum_av, ave_av]
                y_vals = []
                for colour in range(len(cols)):
                    times, ws, hs = [], [], []
                    ws, hs, times = GetWidthHeight(exposure_time, mini, times, ws, hs, f, grad, y_int, imgs[colour], img_avs[colour])
                    ws = np.array([float(x) for x in ws])
                    hs = np.array([float(x) for x in hs])
                    times = np.array(times)

                    if cols[colour] == 'r':
                        ax1.set_title("Red")
                        ax1.scatter(times, hs, s=0.5, label='intensity', c='b')
                        y_vals.append(hs)
                    elif cols[colour] == 'b':
                        ax2.set_title("Blue")
                        ax2.scatter(times, hs, s=0.5, label='intensity', c='b')
                        y_vals.append(hs)
                    elif cols[colour] == 'g':
                        ax3.set_title("Green")
                        ax3.scatter(times, np.array(hs).astype(float), s=0.5, label='intensity', c='b')
                        y_vals.append(hs)
                    elif cols[colour] == 'magenta':
                        ax4.set_title("Sum")
                        ax4.scatter(times, np.array(hs).astype(float), s=0.5, label='intensity', c='b')
                        y_vals.append(hs)
                    elif cols[colour] == 'orange':
                        ax5.set_title("Average")
                        ax5.scatter(times, np.array(hs).astype(float), s=0.5, label='intensity', c='b')
                        y_vals.append(hs)
            except:
                print("f")
        ax1 = legend_without_duplicate_labels(ax1)
        ax2 = legend_without_duplicate_labels(ax2)
        ax3 = legend_without_duplicate_labels(ax3)
        ax4 = legend_without_duplicate_labels(ax4)
        ax5 = legend_without_duplicate_labels(ax5)
        fig.suptitle(mini['Satellite'].iloc[0])
        plt.show()

