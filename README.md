# STM_Final
Final Code &amp; Docs for STM Satellite Identification Project

To implement into Cuillin pipeline:

To run the code, the identification.py file needs only to be ran from the terminal/equivalent with four parameters, e.g: 

    $ python identification.py "2022-05-28" "/home/s1901554/Documents/SpaceTrafficManagement/" "/Images" "/Fits"
    
The parameters are:

- date to be processed
- directory where folders of dates are stored
- path from within date folder to .png images folder
- path from within date folder to .fits folder

See Section 5.1 of the documentation for details on the parameters and more information on running the file.

Prior to this, a few variables (.NEF image size, exposure time, camera position) can optionally be configured in the identification_settings.py file (mainly for if a different specification camera or location is used)

__Note:__ Test data here based on prior code pipeline before bug fix for undetected streaklets due to streaklet length-to-width ratio being too high. Different results may be obtained if the original, un-processed data from the 28th May 2022 is re-processed by the updated code pipeline.
