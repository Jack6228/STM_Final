import os, pandas as pd, numpy as np

# Set directory to where all of the output files from identification.py are stored
a = os.listdir("/home/s1901554/Documents/SpaceTrafficManagement")
b = pd.DataFrame()
dates = []
for i in a:
    if "output_data_" in i and ".csv" in i:
        a = pd.read_csv("/home/s1901554/Documents/SpaceTrafficManagement/"+i)
        a['Date'] = i[12:-4]
        dates.append(a['Date'][0])
        b = pd.concat([b,a])

# Option to get dataframes of satellites which were or were not identified
# fails = b[b['Satellite'].astype(str)=="FAILED"].reset_index(drop=True)
# successes = b[b['Satellite'].astype(str)!="FAILED"].reset_index(drop=True)
start = min(successes['Date'])
end = max(successes['Date'])
b.to_csv("Cumulative_output_data_{}_to_{}.csv".format(start,end))
