import numpy as np
import pandas as pd
import time
###PROCESSING FOR BOSCH DATA
##Important column names
col_names = ['Date_Time','Operating_status:_Flame','Number_of_burner_starts','Operating_status:_Central_heating_active','Operating_status:_Hot_water_active'\
,'Supply_temperature_(primary_flow_temperature)_setpoint'\
,'Supply_temperature_(primary_flow_temperature)','CH_pump_modulation'\
,'Hot_water_outlet_temperature','Hot_water_temperature_setpoint'\
]

## Getting the data
dfOrig = pd.read_csv('2029717969629609985.csv', parse_dates=[['Date','Time']], delimiter='\t')#,nrows=99999)
dfOrig2 = dfOrig[col_names]
dfOrig2['Date_Time'] = pd.to_datetime(dfOrig2['Date_Time'],format='%d.%m.%Y %H:%M:%S',  coerce = True)
dfOrig2 = dfOrig2.fillna(method = 'ffill')
dfOrig2 = dfOrig2.iloc[100:,:]
startTime = dfOrig2.iloc[10,:]['Date_Time']
newDf = pd.DataFrame()
counter = 0
start_t = time.time()

##final "nice" data  gonna be  stored here

# Data = np.zeros((100000,dfOrig2.shape[1]-1))
# time_stamps = []
# while (1==1):
#    prevTime = startTime - pd.Timedelta('30 minutes') ##consider 30 min intervals
#    ##hopefully there is a reading in every 30 minute interval
#    dfPos = dfOrig2[(dfOrig2['Date_Time'] >= prevTime) & (dfOrig2['Date_Time'] < startTime)]
#    if counter%100 == 0:
#       print counter   ##Just a print statement to keep track of how much is done
#    if counter > 100000 or dfPos.shape[0] == 0:##Break at the end if more than 100000 minutes or we have nothing more to add.
#       print "it took", time.time() - start_t 
#       break
#    Data[counter,:] = np.array(dfPos.iloc[-1,1:]) ##Take the last row
#    time_stamps.append(dfPos.iloc[-1,0])
#    ##Update the start time
#    startTime += pd.Timedelta('1 minute')
#    counter += 1 ##Track how many rows are added
# Data = Data[:counter,:]

Data = np.zeros((dfOrig2.shape[0],dfOrig2.shape[1] -1 ))
final_timeStamps = []
##initialize temp data & time
temp_timeStamps = []
temp_Data = []
temp_timeStamps.append((dfOrig2.iloc[0,0]))
temp_Data.append(dfOrig2.iloc[0,1:])

curr_time = dfOrig2.iloc[0,0]
Data[0,:] = np.array(dfOrig2.iloc[0,1:])
counter = 1
i = 1
start_t = time.time()
while i < dfOrig2.shape[0]:
	next_time = dfOrig2.iloc[i,0]
	if next_time - curr_time > pd.Timedelta('1  minute'):
		##pop the top most from temp_timeStamps & temp_Data
		out_data = temp_Data.pop()
		out_time = temp_timeStamps.pop()
		##append the closest to the final list of time
		Data[counter,:] = np.array(out_data)
		final_timeStamps.append(out_time)
		##empty the temp lists
		temp_Data = []
		temp_timeStamps = []
		##add the last one back in
		temp_Data.append(out_data)
		temp_timeStamps.append(out_time)
		##update counter and curr_time
		curr_time += pd.Timedelta('1 minute')
		counter += 1	
	else:
		##just append stuff to the temp - data & timestamps
		temp_Data.append(dfOrig2.iloc[i,1:])
		temp_timeStamps.append(dfOrig2.iloc[i,0])
		i += 1

Data = Data[:counter,:]
print "shape of Data is :", Data.shape
print "counter is;", counter
print "last added is:", Data[counter-1,:]
end_t = time.time()
print "it took:", time.time() - start_t








np.savetxt("cleaned_bosch.csv", Data, delimiter = ",")
np.savetxt("cleaned_bosch_time_stamps.csv",time_stamps,delimiter= ",")