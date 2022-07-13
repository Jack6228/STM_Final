import os, pandas as pd, numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, date

a = os.listdir("/home/s1901554/Documents/SpaceTrafficManagement")
b = pd.DataFrame()
dates = []
for i in a:
    if "output_data_" in i and ".csv" in i:
        a = pd.read_csv("/home/s1901554/Documents/SpaceTrafficManagement/"+i)
        a['Date'] = i[12:-4]
        dates.append(a['Date'][0])

        b = pd.concat([b,a])

fails = b[b['Satellite'].astype(str)=="FAILED"].reset_index(drop=True)
successes = b[b['Satellite'].astype(str)!="FAILED"].reset_index(drop=True)
start = min(successes['Date'])
end = max(successes['Date'])
# b.to_csv("Cumulative_output_data_{}_to_{}.csv".format(start,end))
# exit()

dates = np.array(dates, dtype=str)
filled_date_list = np.arange(start,end,dtype='datetime64[D]')

fig, (ax1, ax2) = plt.subplots(1, 2)
# dates_sorted = sorted(filled_date_list)
# dates_sorted = (sorted(filled_date_list.astype(str), key=lambda x: datetime.strptime(x, '%Y-%m-%d')))

rockets, debris, satels, indices, success_rate, total_obs, unident = [], [], [], [], [], [], []
for d in filled_date_list:
    mini = successes[successes['Date'].astype(str) == str(d)]
    print(d,"--",len(mini))
    mini_fail = fails[fails['Date'].astype(str) == str(d)]
    total_obs.append(len(mini)+len(mini_fail))
    unident.append(len(mini_fail))
    if len(mini) + len(mini_fail) > 0:
        success_rate.append(100*len(mini)/(len(mini)+len(mini_fail)))
    else:
        success_rate.append(np.nan)
    if len(mini) > 0:
        rockets.append(len([x for x in mini['Satellite'].to_list() if "R/B" in x]))
        debris.append(len([x for x in mini['Satellite'].to_list() if " DEB" in x]))
        satels.append(len([x for x in mini['Satellite'].to_list() if "R/B" not in x and " DEB" not in x]))
    else:
        rockets.append(0)
        debris.append(0)
        satels.append(0)
indices = np.arange(len(filled_date_list))
rockets = np.array(rockets)
debris = np.array(debris)
satels = np.array(satels)
total_obs = np.array(total_obs)
unident = np.array(unident)
# total = np.sum(np.vstack((rockets,debris,satels)), axis=0)
ax1b = ax1.twinx()
ax1.bar(indices, rockets, color='b',label='R/Bodies')
ax1.bar(indices, debris, bottom=rockets, color='g', label='Debris')
ax1.bar(indices, satels, bottom=rockets+debris, color='r', label='Satellites')
ax1.bar(indices, unident, bottom=rockets+debris+satels, color='grey', label='Unidentified')
ax1b.scatter(indices, success_rate, marker='x', c='k')
ax1.set_ylabel('Number of Objects [Bars]')
ax1b.set_ylabel('Identification Success Rate [Points]')
ax1.set_xticks(np.arange(len(filled_date_list)))
ax1.set_xticklabels(filled_date_list, rotation=90, fontweight='light')
ax1.set_title("Nightly")
ax1.legend()


names = successes['Satellite'].to_list()
rbs, debs, sats = 0, 0, 0
for n in names:
    if "R/B" in n: rbs +=1
    elif " DEB" in n: debs += 1
    else: sats +=1

vals = [rbs,debs,sats,np.sum(unident)]
nam = ['R/B','DEB','Sat','Fail']
ax2.bar(nam,vals,color=['b','g','r','grey'])
ax2.set_title("Cumulative")
fig.suptitle("Identified Object Distribution {} - {}".format(start,end))
plt.show()

from astral import LocationInfo
from astral.sun import sun
city = LocationInfo("Edinburgh", "Scotland", "Europe/London", 55.923056, -3.187778)
dark_seconds = []
for d in filled_date_list:
    s = sun(city.observer, date=date(*[int(x) for x in str(d).split("-")]))
    dark_seconds.append((s['sunset']-s['sunrise']).total_seconds()/3600)

print(dark_seconds)
fig, ax = plt.subplots(1)
# ax1 = ax.twinx()
# ax.scatter(dark_seconds, np.arange(len(dark_seconds)))
inds = np.where(total_obs != 0)[0]
ax.scatter(np.array(dark_seconds)[inds], total_obs[inds])
ax.set_xlabel("Dark Time (hrs)")
ax.set_ylabel("Satellites Observed")
plt.show()