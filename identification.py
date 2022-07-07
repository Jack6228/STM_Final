# 'Trail' is sometimes used in place of 'streaklet' but both have the same meaning in this code

from os import listdir, remove
from sys import argv
import pandas as pd
from scipy.stats import norm
from numpy import unique, array, sqrt, allclose, arange, where, vstack, average, around
from warnings import catch_warnings, simplefilter, filterwarnings
from skyfield.api import load, wgs84, utc, EarthSatellite
from math import degrees, atan
from datetime import datetime, timedelta
import astropy.units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.io import fits
from astropy.wcs import WCS, utils
from spacetrack import SpaceTrackClient
from identification_settings import *

# Mutes pandas' copy of dataframe outputs to clean run logs
pd.options.mode.chained_assignment = None
# Also mutes a warning from skyfield about invalid sum when calculating an arctan sum - this just indicates that a
#    satellite match it was evaluating could not be performed and thus the catalogued object is not a match -
#    thus the error can be safely ignored to clean the run logs
filterwarnings('ignore', '.*invalid value encountered in remainder.*', )


class DisplayFigure(object):
    """
    Code for displaying figures & images of satellites and detections
    """

    def legend_without_duplicate_labels(self, ax):
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

    def PlotName(self, name, x, y, ax, identified, oo):
        """
        Plots text of a satellite name onto the axis
        :param name: satellite/object name
        :param x, y: x,y position to place text
        :param ax: axis to be used
        :return ax: axis without duplicate legend entries returned
        """
        for i in range(len(x)):
            string = name+" - "+str(round(oo[oo['Name']==name]['Prob'].values[0],1))+"%"
            if name == identified:
                ax.text(x[0]-15,y[0]+30,string,c='magenta',wrap=True,horizontalalignment='right')
            else:
                ax.text(x[0]-15,y[0]+30,string,c='grey',wrap=True,horizontalalignment='right')
            return ax

    def PlotPoints(self, ax, x, y, w, name, identified, streaklet_data, width, height, oo):
        """
        Plot dots onto an axis to represent satellite events
        :param ax: axis to plot onto
        :param RaDec: array of x and y values to plot
        :param w: wcs transformation of file
        :param name: name of satellite
        :return ax: updates axis
        """
        if name == identified:
            ax.plot(x,y, '-', c='magenta', alpha=0.25)
        else:
            ax.plot(x,y, '-', c='grey', alpha=0.25)
        self.PlotName(name, x, y, ax, identified, oo)
        # Only plots start and end points if more than one point; if not, replots the line as a point so it's visible
        if len(x) > 1:
            ax.plot(x[0],y[0],   '^', c='cyan', alpha=0.25, markersize=7, label="start")
            ax.plot(x[-1],y[-1], 'v', c='cyan', alpha=0.25, markersize=7, label='end')
        else:
            ax.plot(x,y, 'o', c='cyan', alpha=0.25, markersize=5)

        #sat_x = [streaklet_data[3], streaklet_data[7]]
        #sat_y = [streaklet_data[4], streaklet_data[8]]
        #ax.plot(sat_x[0],sat_y[0],   '^', c='magenta', alpha=0.25, markersize=7, label="start")
        #ax.plot(sat_x[1],sat_y[1], 'v', c='magenta', alpha=0.25, markersize=7, label='end')
        return ax

    def LoadJPGImage(self, arr, w, width, height, streaklet_data, name, oo):
        """
        Sets up the display axes, loads the image from a file and sets up the axes of the plot in RA and Dec from wcs
        :param arr: dataframe of [name, gradient, y-int, x, y] of each satellite
        :param w: wcs transformation of file
        :param width, height: width and height of image (from wcs)
        :param streaklet_data: list of data from the streaks txt file of the data of each streak in an image
        :return ax: returns updated axis
        """
        ax = plt.subplot(111, projection=w)
        image = plt.imread(self.image_path+streaklet_data[0][:-3]+'png')#, format=streaklet_data[0][-3:])
        # This image is flipped in the y-axis
        ax.imshow(image)
        ax.set_xlabel("Right Ascension [hrs]")
        ax.set_ylabel("Declination [deg]")
        ax.coords.grid(color='white', alpha=0.25, linestyle='solid')
        # overlay = ax.get_coords_overlay('fk5')
        # overlay['ra'].set_ticks(color='white')
        # overlay['dec'].set_ticks(color='white')
        # overlay['ra'].set_axislabel('Right Ascension [deg]')
        # overlay['dec'].set_axislabel('Declination [deg]')
        # overlay.grid(color='white', linestyle='solid', alpha=0.25)
        # Plots white x at center of grid
        ax.plot(width/2,height/2, 'x', c='white', label='center')
        ax.plot(0,0, 'x', c='cyan', label='center')
        # Plots all points for this trail
        for i in range(len(arr)):
            self.PlotPoints(ax, arr.iloc[i,1], arr.iloc[i,2], w, arr.iloc[i,0], name, streaklet_data, width, height, oo)
        return ax

    def Plot(self, arr, w, width, height, trail_row, path, name, xx, filename, oo, cutoff):
        """
        Function to plot an image of overlaid trails and co-ordinate grid
        :param arr: dataframe of [name, gradient, y-int, x, y] of each satellite
        :param w: wcs transformation of file
        :param width, height: width and height of image (from wcs)
        :param trail_row: list of data from the streaks txt file of the data of each streak in an image
        :param path: location where files are stored (set in IdentifySatellites __init__)
        """
        self.path = path
        self.image_path = self.path+"/Images/"
        ax = self.LoadJPGImage(arr, w, width, height, trail_row, name, oo)
        plt.gca().invert_yaxis()
        objects = self.legend_without_duplicate_labels(ax)
        ax.patch.set_facecolor('black')
        plt.title(filename)
        if cutoff == True: plt.savefig("#"+str(xx+1)+"_cutoff.png",facecolor='w')
        else: plt.savefig("#"+str(xx+1)+".png",facecolor='w')

class IdentifySatellites(object):
    """
    Code to process an image and identify its satellite trails
    Run() is first function to be called
    """

    def __init__(self, date, path, img_path, wcs_path):
        """
        Initialises main variables for running
        """
        print("---- Initialising variables...")
        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        # User defined variables - will be linked to settings_template.py when integrated to Cuillin (from settings import *)
        self.date = date # date of images to be processed
        self.path = path # path to folder (e.g. in khaba) where this date's folder is stored
        self.image_path = img_path# change folder names to image folder (detected_streaks) and wcs (wcs) folder within self.path folder
        self.wcs_path = wcs_path
        # ----------------------------------------------------------------------------------------------------
        # Variables which are constant for the ROE camera but the option to change them here (i.e. if camera was different)
        self.exposure_time = is_exposure_time
        self.nef_h, self.nef_w = is_nef_h, is_nef_w
        lat, long, elevation = is_lat, is_long, is_elevation
        self.spacetrack_email = is_spacetrack_email
        self.spacetrack_password = is_spacetrack_password
        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------

        # Dataframe into which results of all streaklet identification attempts will be stored
        self.output_file = []

        # Loads streak_data.txt file to get positions and times of streaklet start & end points
        self.ProcessStreaksTXT()

        # Initialises skyfield variables from observing position
        self.ts = load.timescale()
        self.bluffton = wgs84.latlon(lat,long,elevation)

        # Gets TLEs from catalog at space-track.org
        self.GetTLEs()

        print("---- Completed set-up --\n")

    def ProcessStreaksTXT(self):
        """
        Loads variables for data of positions, timing, filenames etc of each streaklet form txt file
        """
        # Reads a txt file of the start and end of every streaklet across all images that were processed, adds header names and sorts by filename (chronologically)
        self.streaklets = pd.read_csv(self.path+'/streaks_data.txt', sep=",", header=None)
        self.streaklets.columns = ['Filename','RA1','Dec1','x1','y1','RA2','Dec2','x2','y2']
        self.streaklets = self.streaklets.sort_values(by=['Filename']).reset_index(drop=True)

        # Processed streaks_data.txt file doesn't have streak_x identification number in the png filename column, so adds it in here by ordering and changing filename
        filenames = self.streaklets['Filename'].to_numpy()
        filenames_unique, fn_counts = unique(filenames, return_counts=True)
        offset, iter = 0, 0
        while iter < len(fn_counts):
            val = fn_counts[iter]
            for k in range(val):
                x = iter + k + offset
                self.streaklets.at[x, 'Filename'] = str(self.streaklets['Filename'][x].replace('.NEF','_streak_'+str(k+1)+'.png'))
            offset += val - 1
            iter += 1

        print("       Total number of streaklets detected on this night:",len(self.streaklets))

    def GetTLEs(self):
        """
        #     Uses the date provided to either load or download a catalog of objects from space-track.org depending on if it has already been downloaded
        """
        # Gets list of files in working directory, creates datetime object of date of data to be processed, and forms empty dataframe of catalogues from space-track
        files = listdir()
        date = datetime(*[int(x) for x in self.date.split("-")])
        tle_data = pd.DataFrame()

        # Loops back x days into the past from the above date to download catalogues from space-track
        days_back = 14
        print("       Loading space-track catalogues:")
        for i in range(-1,days_back):
            # Builds a string representing the date range of the query to be sent to space-track's API
            #     Format: YYYY-mm-dd--YYYY-mm-dd
            this_date = date - timedelta(days = i+1)
            this_date_end = date - timedelta(days = i)
            this_date_list = [this_date.year, this_date.month, this_date.day]
            this_date_list_end = [this_date_end.year, this_date_end.month, this_date_end.day]
            date_query = "-".join([str(x).zfill(2) for x in this_date_list])+"--"+"-".join([str(x).zfill(2) for x in this_date_list_end])

            # Loops through all the files found in the working directory to determine if a catalog has already been downloaded for the given date range
            cat_file = []
            for f in files:
                if date_query in f and "ST_Catalog" in f:
                    cat_file.append(f)

            if cat_file != []:
                # If found an existing catalog, loads and appends to tle_data
                print("         Found pre-existing catalogue from Space-Track.org      {}".format(date_query))
                tle_data = pd.concat([tle_data, pd.read_csv("ST_Catalog_"+date_query+".csv", keep_default_na=False)])
            else:
                # Otherwise uses the spacetrack package to download data from space-track
                print("         API-accessing space-track.org data for this date     {}".format(date_query))
                st = SpaceTrackClient(identity=self.spacetrack_email, password=self.spacetrack_password)
                data = st.gp_history(creation_date=date_query,format='csv')

                # Formats data correctly so it can be stored in a pandas dataframe
                data = data.replace('"', '')
                # [:-1] removes final element of list since the csv string ends with \n. [1:] separates headers into actual headers of df
                cat = pd.DataFrame([x.split(',') for x in data.split('\n')[1:-1]], columns=[x for x in data.split('\n')[0].split(',')])
                # Saves dataframe to file for future use and appends to tle_data
                cat.to_csv("ST_Catalog_"+date_query+".csv", index=False)
                tle_data = pd.concat([tle_data, pd.read_csv("ST_Catalog_"+date_query+".csv", keep_default_na=False)]).reset_index(drop=True)

        print("      ## Catalogues loaded")

        # Deletes previously downloaded catalogues for dates which are greater than 21 days in the past from the current date being processed
        date_to_delete_before = date - timedelta(days = 21)
        print("      Purging old catalogues:")
        for f in files:
            if "ST_Catalog_" in f:
                file_date = datetime.strptime(f[11:21], "%Y-%m-%d")
                if file_date < date_to_delete_before:
                    print("        {}".format(f))
                    remove(f)
        print("      ## Catalogues purged where neccessary")


        # Removes objects from the catalogue which have already decayed from orbit
        tle_data = tle_data[tle_data['DECAY_DATE']=='']
        # Get epoch of each TLE in datetime format and save as another column: 'Epoch_Datetime'
        tle_data['Epoch_Datetime'] = pd.to_datetime(tle_data['EPOCH'])
        old_len = len(tle_data)

        # Cuts catalog to find the instance of each satellite which has a TLE epoch closest to midnight of the night of observation
        time = [int(x) for x in self.date.split("-")]
        tle_data = self.CutCatalogByEpoch(datetime(*time,0,0,0)+timedelta(days=1), tle_data)

        # Displays how many unique objects are in the catalogue compared to that number including duplicates
        print("       Optimised the LEO TLE catalog from {} to {} objects".format(int(old_len),len(tle_data)))

        # Loops through this catalog to find satellite objects (from skyfield package) for each satellite and adds this & other data to a new dataframe
        self.catalogue = []
        for i in range(len(tle_data)):
            sat_obj = EarthSatellite(tle_data['TLE_LINE1'].iloc[i], tle_data['TLE_LINE2'].iloc[i], tle_data['OBJECT_NAME'].iloc[i], self.ts)
            self.catalogue.append([sat_obj, tle_data['NORAD_CAT_ID'].iloc[i], tle_data['Epoch_Datetime'].iloc[i], tle_data['TLE_LINE1'].iloc[i], tle_data['TLE_LINE2'].iloc[i], tle_data['OBJECT_NAME'].iloc[i]])
        self.catalogue = pd.DataFrame(data=self.catalogue, columns=["SatelliteObject","NORAD_CAT_ID","Epoch_Datetime","TLE1","TLE2","SatName"])

    def CutCatalogByEpoch(self, time, catalog):
        """
        From a given time at which an image was taken, goes through the space-track catalog and only keeps one
            instance of each unique satellite - the one with the closest epoch to the image capture time
        """
        # Goes through all duplicate rows in a catalog by NORAD_CAT_ID (unique satellite identifier) and gets the one with the closest epoch
        # Calculates time difference between TLE epoch and image capture time
        catalog['EpochDiff'] = abs(catalog['Epoch_Datetime']-time)
        # Sorts by satellite ID and 'EpochDiff' and then takes the first row for each satellite ID (smallest EpochDiff)
        catalog = catalog.sort_values(by=['NORAD_CAT_ID','EpochDiff'])
        catalog = catalog.drop_duplicates(subset=['NORAD_CAT_ID'], keep='first').reset_index(drop=True)
        return catalog

    def ProcessImage(self, streaklet_data):
        """
        Uses the obtained wcs fits file (from astrometry.net) to get main info about an image and generate a transformation object to convert between pixels and RA Dec
        Then uses skyfield to find all of the satellites which pass through the camera's FOV in the time interval of this image
        """
        # Loads the wcs.fits file and gets bounds & times of images etc.
        wcs_filename = streaklet_data[0].replace("streak","wcs")
        f = fits.open(self.wcs_path+'/'+wcs_filename[:-4]+'.fits')

        # Prevents printing error message about 'WCS transformation has more axes (2)...'
        with catch_warnings():
                simplefilter('ignore')
                w = WCS(f[0].header, naxis=2)

        # Gets properties about the image such as pixel height & width, time of capture from the .fits header
        header = f[0].header
        width = header[20]
        height = header[21]

        # Converts date of image capture to a list of [YYYY, MM, DD]
        date = list(map(int, streaklet_data[9].strftime('%Y,%m,%d').split(",")))
        # Lists of [YYYY, MM, DD, HH, MM, SS] for start and end times of shutter being open
        start = date+list(map(int, streaklet_data[9].strftime('%H,%M,%S').split(",")))
        end = date+list(map(int, streaklet_data[10].strftime('%H,%M,%S').split(",")))
        # Optional parameter to allow for a fudge factor or offset in the shutter trigger
        shutter_offset = 0#1.85

        # Creates a range of times from start to end of exposure of this image at 1 second intervals
        seconds_range = arange(start[-1]+shutter_offset, start[-1]+(datetime(*end)-datetime(*start)).total_seconds()+1+shutter_offset)
        intervals = self.ts.utc(*start[:5],seconds_range)

        # Co-ordinates of start & end of trail in RA Dec (don't reflect which point came first in time)
        one = [streaklet_data['RA1'],streaklet_data['Dec1']]
        two = [streaklet_data['RA2'],streaklet_data['Dec2']]

        # Creates some lengths by which to form a frame in which a satellite must be to be determined as 'in the FOV of the camera'
        #     These are based on the length of the streaklet
        ra_diff = abs(two[0]-one[0])/2
        dec_diff = abs(two[1]-one[1])/2
        mid_ra_dec = [abs(two[0]-one[0])/2+min(two[0],one[0]),abs(two[1]-one[1])/2+min(two[1],one[1])]

        # Gets events of each satellite for this image
        satellites_in_image = []
        for s in range(len(self.catalogue)):
            # Skyfield calculations to find vectors between satellite and observer
            difference = self.catalogue['SatelliteObject'][s] - self.bluffton
            # Gets topocentric co-ords at each timepoint given
            topocentric = difference.at(intervals)
            # Finds RA and Dec values at each timestep (& converts to degrees)
            ras, decs, dists = topocentric.radec()
            ras = ras._degrees
            decs = decs._degrees

            # these if statements are perhaps the most inefficient part of the code
            # Test if this satellite was in the camera's FOV in this time
            if len(where((mid_ra_dec[0]-(ra_diff*1)  <= ras) & (ras <= mid_ra_dec[0]+(ra_diff*1)))[0]) > 0:
                if len(where((mid_ra_dec[1]-(dec_diff*1)  <= decs) & (decs <= mid_ra_dec[1]+(dec_diff*1)))[0]) > 0:
                    # If it is, create an astropy SkyCoord object to be able to convert between RA Dec and pixels
                    # Then converts the RA Decs to pixels (x,y) and forms an array of these (skycoord_to_pixel) using the wcs 'w' variable from earlier
                    x, y = utils.skycoord_to_pixel(SkyCoord(ras, decs, frame=FK5, unit=(u.deg, u.deg)), w, 0, mode='wcs')
                    xys = vstack((x,y)).T
                    # Appends all this data to dataframe
                    satellites_in_image.append([self.catalogue['SatelliteObject'][s].name, xys[:,0], xys[:,1], self.catalogue['NORAD_CAT_ID'][s], self.catalogue['TLE1'][s], self.catalogue['TLE2'][s], [ras[0],decs[0]], [ras[-1],decs[-1]]])
        satellites_in_image = pd.DataFrame(satellites_in_image,columns=['Name','x','y','NORAD_CAT_ID','TLE1','TLE2','RADecPt1','RADecPt2'])
        print("    {} satellite(s) identified in image to test against".format(len(satellites_in_image)))
        return w, width, height, satellites_in_image, one, two

    def AnalyseMatches(self, arr, i, width, height, cutoff, RADecPoint1, RADecPoint2):
        """
        For each satellite identified in an image, this function determines which one is the most likely match based on trail length, angle and distance from expected position (from TLE)
        """
        # Satellite x and y pixel positions (observed)
        sat_x = [self.streaklets['x1'][i], self.streaklets['x2'][i]]
        sat_y = [self.streaklets['y1'][i], self.streaklets['y2'][i]]
        # Pixel co-ord of centre of image
        mid_image = [width/2,height/2]
        # Expected pixel x and y points of satellite (from skyfield)
        test_x = arr['x'].to_numpy()
        test_y = arr['y'].to_numpy()

        # Simple arrays of satellite names, IDs, TLEs
        c1 = arr['Name'].to_numpy()
        c2 = arr['NORAD_CAT_ID'].to_numpy()
        c3 = arr['TLE1'].to_numpy()
        c4 = arr['TLE2'].to_numpy()

        # Creates a new array of the first and last points of the trail (i.e. ignores those points inbetween)
        for i in range(len(test_x)):
            if len(test_x[i]) != 1:
                test_x[i] = array([test_x[i][0], test_x[i][-1]])
            if len(test_y[i]) != 1:
                test_y[i] = array([test_y[i][0], test_y[i][-1]])
        test_x = array(test_x)
        test_y = array(test_y)

        # Length of streaklet (observed)
        length = sqrt((sat_x[1]-sat_x[0])**2 + (sat_y[1]-sat_y[0])**2)

        matches = []
        # Analysing match likelihood by distance between observed and expected positions ---------------------------------------------
        # Creates normal distribution of distance (mean of 0 pixel separation, sigma of 1/4 of streaklet length)
        distribution = norm(loc=0,scale=length/4)
        dists = []
        # Get PDF of distance compared to distribution for each possible match
        for i in range(len(test_x)):
            center = [abs(test_x[i][0]-test_x[i][1])/2+min(test_x[i][0],test_x[i][1]),abs(test_y[i][0]-test_y[i][1])/2+min(test_y[i][0],test_y[i][1])] # center of satellite trail being overlaid from catalog
            dist = sqrt((center[0]-mid_image[0])**2 + (center[1]-mid_image[1])**2)
            dists.append((distribution.pdf(dist)/distribution.pdf(0))*100)
        matches.append(dists)

        # Now by streaklet lengths ---------------------------------------------------------------------------------------------------
        # Creates normal distributions of lengths (mean of observed length, sigma of 1/10 of mean)
        distribution = norm(loc=length,scale=length*0.1)
        # Second distribution for streaklets that are cut-off by image bounds which has a sigma = mean
        distribution_cutoff = norm(loc=length,scale=length)
        lengths = []
        for i in range(len(test_x)):
            test_len = sqrt((test_x[i][1]-test_x[i][0])**2 + (test_y[i][1]-test_y[i][0])**2)
            # If trail is fully shown in image, get normal PDF, if not then use a distribution with stdev of cut trail's length, not 10%
            if cutoff == False:
                lengths.append((distribution.pdf(test_len)/distribution.pdf(length))*100)
            else:
                lengths.append((distribution_cutoff.pdf(test_len)/distribution_cutoff.pdf(length))*100)
        matches.append(lengths)

        # Finally by streaklet angles ------------------------------------------------------------------------------------------------
        # Calculates observed angle of streaklet (from x,y pixel reference frame)
        angle = degrees( atan( abs(sat_y[0]-sat_y[1])/abs(sat_x[0]-sat_x[1]) ) )
        # Creates normal distribution of angles (mean of observed angle, sigma of 10 deg)
        distribution = norm(loc=angle,scale=10)
        angles = []
        for i in range(len(test_x)):
            ang = degrees( atan( abs(test_y[i][1]-test_y[i][0])/abs(test_x[i][1]-test_x[i][0]) ) )
            angles.append((distribution.pdf(ang)/distribution.pdf(angle))*100)
        matches.append(angles)

        # Orders points by direction of streak where possible
        trail1 = RADecPoint1
        trail2 = RADecPoint2
        for i in range(len(test_x)):
            cat1 = array(arr['RADecPt1'][i])
            cat2 = array(arr['RADecPt2'][i])
            ang = degrees( atan( (cat1[1]-cat2[1]) / (cat1[0]-cat2[0]) ) )
            ang2 = degrees( atan( (cat2[1]-cat1[1]) / (cat1[0]-cat2[0]) ) )
            if ang/ang != angle/angle:
                point1 = [trail1[0], trail1[1]]
                point2 = [trail2[0], trail2[1]]
            else:
                point1 = [trail2[0], trail2[1]]
                point2 = [trail1[0], trail1[1]]

        # Combining all three values to one probability
        matches = array(matches)
        means = average(matches, axis=0)
        results = pd.DataFrame(data=vstack((c1,c2,means,matches,c3,c4)).T,columns=['Name','NORAD_CAT_ID','Prob','Distance','Length','Angle','TLE1','TLE2'])
        # Sorts results so best match is in top row of dataframe
        results = results.groupby('Name').max().sort_values(by=['Prob'], ascending=False).reset_index()

        # Returns parameters to save to file
        return results, point1, point2

    def IsTrailCutOff(self, i):
        """
        Determines if a streaklet is cut-off by the edge of the image
        """
        sat_x = array([self.streaklets['x1'][i], self.streaklets['x2'][i]])
        sat_y = array([self.streaklets['y1'][i], self.streaklets['y2'][i]])
        if max(sat_x) > self.nef_w or max(sat_x) < 0 or max(sat_y) > self.nef_h or max(sat_y) < 0:
            print("    The trail in this image is cut off by the bounds of the .NEF image")
            return True
        else: return False

    def RunIdentification(self, num_identifications):
        """
        Runs identification process:
            Processes image and determines which satellite a trail is likely to be
        """
        # Empty dataframe for failed identifications
        failed = []
        for i in range(len(self.streaklets)):

            print("\nStreaklet No. {}/{}:".format(i+1,len(self.streaklets)),"\n  File:",self.streaklets["Filename"][i],"\n  Time:",self.streaklets["Time1"][i],"UTC/BST-1")
            # Process image to get points of start and end of trail etc. and list of possible matches ('arr')
            w, width, height, arr, RADecPoint1, RADecPoint2 = self.ProcessImage(self.streaklets.iloc[i])
            # Determines if trail is cut off by image bounds
            cutoff = self.IsTrailCutOff(i)

            # If satellites are found to have been in this FOV at this time, see which is best match through AnalyseMatches()
            if len(arr) > 0:
                results, point1, point2 = self.AnalyseMatches(arr, i, width, height, cutoff, RADecPoint1, RADecPoint2)
                name, id, certainty = results['Name'][0], results['NORAD_CAT_ID'][0], results['Prob'][0]
                num_identifications += 1
                print("  {} ({}) -- with {}%".format(name, id, round(certainty,1)))
                # Append this satellite to the output file dataframe
                self.output_file.append([i+1        # Number of loop
                    ,self.streaklets["Filename"][i]          # Filename
                    ,name                           # Satellite name
                    ,id                             # Satellite ID
                    ,round(certainty,2)             # Likelihood
                    ,round(results['Distance'][0])
                    ,round(results['Length'][0])
                    ,round(results['Angle'][0])
                    ,cutoff                         # True / False
                    ,around(point1, 4)                         # First point (in time)
                    ,around(point2, 4)                         # Second point
                    ,results['TLE1'][0]
                    ,results['TLE2'][0]])

                # # Optional plotting ability
                # DF.Plot(arr, w, width, height, self.streaklets.iloc[i].tolist(), self.path, name, i, self.streaklets["Filename"][i], results, cutoff)

            else:
                print(" --- No satellites in image --- ")
                # Appends streaklets that can't be identified to a separate dataframe
                failed.append(self.streaklets.iloc[i].to_list())

        print("Number of streaklets with possible matches found: {} / {}".format(num_identifications, len(self.streaklets)))

        return num_identifications, failed

    def Run(self):
        """
        __init__() function will have been automatically called prior to this function being run
        Runs program using above functions to determine identity of each trail in all images
        """
        # Create some empty columns and then fill them within the for loop
        self.streaklets["Time1"], self.streaklets["Time2"] = "", ""
        for i in range(len(self.streaklets)):
            self.streaklets["Time1"][i] = datetime.strptime(self.streaklets['Filename'][i][4:21],'%Y-%m-%d_%H%M%S')
            self.streaklets["Time2"][i] = self.streaklets["Time1"][i] + timedelta(seconds=self.exposure_time)

        num_identifications = 0
        # Creates new instance of self.streaklets that can be updated without overwriting original
        inp = self.streaklets

        # Runs identification process
        print("----  {} Identifications To-Do  ----".format(len(inp)))
        num_identifications, fails = self.RunIdentification(num_identifications)
        # The instance of inp that is returned is the dataframe of observed streaklets which could not be identified

        # Creates a dataframe from failed identifications
        fails = pd.DataFrame(data = fails, columns=self.streaklets.columns)
        # Appends non-identifiable streaklets to output file at the end
        for i in range(len(fails)):
            self.output_file.append(["",fails["Filename"][i],"FAILED","","","","","","",[fails['RA1'][i],fails['Dec1'][i]],[fails['RA2'][i],fails['Dec2'][i]],"",""])

        # Saves dataframe of all satellite streaklets and their identification results
        out = pd.DataFrame(data=self.output_file, columns=["Photo#","Filename","Satellite","NORAD_CAT_ID","Likelihood","Distance","Length","Angle","Cutoff","RADecPoint1","RADecPoint2","TLE1","TLE2"])
        out.to_csv("output_data_"+str(self.date)+".csv", index=False)

# Gets input parameters from terminal prompt
# Example terminal input:
#     python identification.py "2022-05-28" "/home/s1901554/Documents/SpaceTrafficManagement/" "/Images" "/Fits"
# See section 5.1 of documentation for in-depth details
if __name__ == "__main__":
    date = str(argv[1])
    path = str(argv[2])+date
    image_path = path+str(argv[3])
    wcs_path = path+str(argv[4])

# Initialises classes
IS = IdentifySatellites(date, path, image_path, wcs_path)
# # Uncomment both if useing DisplayFigure() Class:
# import matplotlib.pyplot as plt
# DF = DisplayFigure()

# Runs identification class through main Run() function
IS.Run()
