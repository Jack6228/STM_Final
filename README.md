# STM_Final
Final Code &amp; Docs for STM Satellite Identification Project

To implement into Cuillin pipeline:

To run the code, the identification.py file needs only to be ran from the terminal/equivalent, e.g: $ python identification.py

Prior to this, a few variables must be configured within the __init__() function of the IdentifySatellites() class:

    Date of images to process

    Path to folder of data for given date (*)

    Path to subfolders in * for .png images and .fits files

    Some other variables can be adjusted, but are constant for all observations taken at the ROE (Earth coordinates, exposure time, .NEF image size)

// These could be put into the settings file from the previous pipeline instead of being defined here if so wished.

Note:: Test data here based on prior code pipeline before bug fix for undetected streaklets due to streaklet length-to-width ratio being too high. Different results may be obtained if the original, un-processed data from the 28th May 2022 is re-processed by the updated code pipeline.
