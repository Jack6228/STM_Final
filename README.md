# STM_Final
Final Code &amp; Docs for STM Satellite Identification Project

How to Use
        
1) To install dependencies:

    $ python3 -m pip install --user pandas
    
    $ python3 -m pip install --user skyfield
    
    $ python3 -m pip install --user astropy
    
    $ python3 -m pip install --user spacetrack

2) Running the code
          
To run this script, only the identification.py file is needed and in the directory where it is located, the code from identification import IdentifySatellites should be run to start the identification process. The terminal command follows the structure of:
        
    $ python identification.py date file_path image_path fits_path
            
where date is a string of the date of images to be processed, e.g:
            
    "2022-05-28"
            
file_path is the directory in which results from every date of observations is stored, e.g:

    "/user/MyDocuments/MyImages/Nights/"

image_path is a sub folder within file_path which contains the cut-out .png images of streaklets, note that the sub-folder identifier "/" is required, e.g:
            
    "/Images"

and fits_path is the same as file_path but for the .fits files corresponding to each streaklet, e.g:

    "/Fits"

3) Checking against test data
          
Test data is provided for the night of the 28th May 2022 (see \href{https://github.com/Jack6228/STM_Final}{GitHub Repository} for data).
If the "2022-05-28" data folder from GitHub is downloaded into the same directory where the \code{identification.py} file is being run from (for example, "/path/to/test/data/") the following prompt should run the script and provide results which can be compared to those on GitHub:
            
    $ python identification.py "2022-05-28" "/path/to/test/data/" "/Images" "/Fits"

The output will be a .csv file titled: benchmark_output_data_2022-05-28.csv
The difference between both files can be compared by:

    $ diff benchmark_output_data_2022-05-28.csv output_data_2022-05-28.csv

See Section 5.1 of the documentation for details on the parameters and more information on running the file.

Prior to this, a few variables (.NEF image size, exposure time, camera position) can optionally be configured in the identification_settings.py file (mainly for if a different specification camera or location is used)

__Note:__ Test data here based on prior code pipeline before bug fix for undetected streaklets due to streaklet length-to-width ratio being too high. Different results may be obtained if the original, un-processed data from the 28th May 2022 is re-processed by the updated code pipeline.
