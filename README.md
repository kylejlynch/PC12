
# Electric Field Growth Dependent Behavior of PC12 Neuron-analogs

- [Introduction](#introduction)
- [Data Extraction and Cleaning](#data-extraction-and-cleaning)
- [Statistics](#statistics)
- [Plotting](#plotting)
- [Conclusion](#conclusion)

[View the Jupyter Notebook Here](https://nbviewer.jupyter.org/github/kylejlynch/PC12/blob/master/PC12_Electric_Field.ipynb)

## Introduction

This notebook will explain a portion of my master's thesis research.
The goal of my research was to develop a way in which to expose neurons growing on the surface of a particular  material to electric fields and compare to neurons that were not subject to such stimulation. Figure 1 below shows the electrical device that I designed and constructed.

![Figure 1. Experimental Design and Electrical Device](https://i.imgur.com/5X2k7fP.png)

Prior to application of the electric field, I first had to coat the surface of the substrate with an extracellular matrix so that the cells would adhere. Once the cells were plated, I connected the electrical device to a voltage source for approximately 24 hrs along side of a control with no voltage.

After the 24 hr electrical stimulation, I fixed the cells for scanning electron microscopy (SEM) and imaged the samples. Imaging would result in dozens of pictures per sample per trial. I then measured neurite length and orientation using ImageJ as shown below in Figure 2. Yellow lines indicate traced neurites for length measurement. Red lines indicate angles of respective neurites to an imaginary 0 degree axis. The anode of the electrical device was located at 270 degrees in all images.

![Figure 2. Neurite Length and Angle Measurement](https://i.imgur.com/Mn2DdBC.png)

Each image measured generated an excel file as shown below in Figure 3. Hundreds on neurites were measured along with a lot of junk data that needed to be cleaned up before analysis could take place.

![Figure 3. Raw Data in Excel](https://i.imgur.com/s6RqHRq.png)

## Data Extraction and Cleaning

The code below extracts the data needed from all files for 3 trials each for both experimental and control conditions.
    
```python
import os
import numpy as np
import pandas as pd

trialdata=[]
def neurimport() :
    listing = os.listdir(path)
    for file in listing :
        handle = open(os.path.join(path,file) , 'r')
        #extract data
        dfraw = pd.read_csv(handle)
        if float(dfraw.loc[ : (dfraw['Angle'].count()/2)-1,
                           ['Angle']].sum()) != 0:
            #delete unwanted data
            dfdata = dfraw.loc[ : (dfraw['Angle'].count()/2)-1, 
                               ['Angle']].reset_index(drop=True)
            dfdata['Length'] = dfraw.loc[(dfraw['Length'].count()/2) : , 
                              ['Length']].reset_index(drop=True)
        #correction in case angles were measured first
        if float(dfraw.loc[ : (dfraw['Angle'].count()/2)-1,
                           ['Angle']].sum()) == 0:
            #delete unwanted data
            dfdata = dfraw.loc[(dfraw['Angle'].count()/2) : , 
                               ['Angle']].reset_index(drop=True)
            dfdata['Length'] = dfraw.loc[ : (dfraw['Length'].count()/2), 
                              ['Length']].reset_index(drop=True)
        #convert negative angles
        dfdata['Angle'] = dfdata['Angle'].apply(lambda x : 
                                                x + 360 if x < 0 else x)
        #conversion to radians
        dfdata['Angle'] = dfdata['Angle'].apply(lambda x : np.radians(x))
        #vector components
        dfdata['Lcos'] = np.cos(dfdata['Angle']) * dfdata['Length']
        dfdata['Lsin'] = np.sin(dfdata['Angle']) * dfdata['Length']
        g = dfdata.groupby(3.1415>dfdata['Angle'])['Length'].mean()
        dflgroup = pd.Series(g.values, index=['Anode','Cathode'])
        trialdata.append(dfdata)

#Trial 1
path = 'Neurite_data/csv/15V PCSA/1/'
a = neurimport()
dftrial = pd.concat(trialdata)
pcsa15V = {
            'Avg Length' : dftrial['Length'].mean(), 
            'Lcos' : dftrial['Lcos'].mean(),
            'Lsin' : dftrial['Lsin'].mean(),
           }
df = pd.DataFrame(pcsa15V, index=['Trial 1'])

trialdata=[]
del dftrial

#Trial 2
path = 'Neurite_data/csv/15V PCSA/2/'
a = neurimport()
dftrial = pd.concat(trialdata)
pcsa15V = {
            'Avg Length' : dftrial['Length'].mean(), 
            'Lcos' : dftrial['Lcos'].mean(),
            'Lsin' : dftrial['Lsin'].mean(),
           }
df.loc['Trial 2'] = pcsa15V

trialdata=[]
del dftrial

#Trial 3
path = 'Neurite_data/csv/15V PCSA/3/'
a = neurimport()
dftrial = pd.concat(trialdata)
pcsa15V = {
            'Avg Length' : dftrial['Length'].mean(), 
            'Lcos' : dftrial['Lcos'].mean(),
            'Lsin' : dftrial['Lsin'].mean(),
           }
df.loc['Trial 3'] = pcsa15V

trialdata=[]
del dftrial

#Trial 1
path = 'Neurite_data/csv/0V PCSA/1/'
a = neurimport()
dftrial = pd.concat(trialdata)
pcsa0V = {
            'Avg Length' : dftrial['Length'].mean(), 
            'Lcos' : dftrial['Lcos'].mean(),
            'Lsin' : dftrial['Lsin'].mean(),
           }
df2 = pd.DataFrame(pcsa0V, index=['Trial 1'])

trialdata=[]
del dftrial

#Trial 2
path = 'Neurite_data/csv/0V PCSA/2/'
a = neurimport()
dftrial = pd.concat(trialdata)
pcsa0V = {
            'Avg Length' : dftrial['Length'].mean(), 
            'Lcos' : dftrial['Lcos'].mean(),
            'Lsin' : dftrial['Lsin'].mean(),
           }
df2.loc['Trial 2'] = pcsa0V

trialdata=[]
del dftrial

#Trial 3
path = 'Neurite_data/csv/0V PCSA/3/'
a = neurimport()
dftrial = pd.concat(trialdata)
pcsa0V = {
            'Avg Length' : dftrial['Length'].mean(), 
            'Lcos' : dftrial['Lcos'].mean(),
            'Lsin' : dftrial['Lsin'].mean(),
           }
df2.loc['Trial 3'] = pcsa0V

trialdata=[]
del dftrial
```

## Statistics

The code below computes averages and standard error in preparation for plotting.

```python
#Statistics
print('===15V PCSA===')
avgpcsa15_length = df['Avg Length'].mean()
pcsa15_sem = df['Avg Length'].sem(ddof=0)
#avgpcsa15_lcathode = df['Lcathode'].mean()
#pcsa15_lcathode_sem = df['Lcathode'].sem(ddof=0)
#avgpcsa15_lanode = df['Lanode'].mean()
#pcsa15_lanode_sem = df['Lanode'].sem(ddof=0)
avgpcsa15_lsin = df['Lsin'].mean()
pcsa15_lsin_sem = df['Lsin'].sem(ddof=0)
avgpcsa15_lcos = df['Lcos'].mean()
pcsa15_lcos_sem = df['Lcos'].sem(ddof=0)
print('Average of',len(df['Avg Length']), 'Trials :', avgpcsa15_length, '+/-',
      pcsa15_sem)
#print('Average cathodal :', avgpcsa15_lcathode, '+/-', pcsa15_lcathode_sem)
#print('Average anodal :', avgpcsa15_lanode, '+/-', pcsa15_lanode_sem)

print('===0V PCSA===')
avgpcsa0_length = df2['Avg Length'].mean()
pcsa0_sem = df2['Avg Length'].sem(ddof=0)
#avgpcsa0_lcathode = df2['Lcathode'].mean()
#pcsa0_lcathode_sem = df2['Lcathode'].sem(ddof=0)
#avgpcsa0_lanode = df2['Lanode'].mean()
#pcsa0_lanode_sem = df2['Lanode'].sem(ddof=0)
avgpcsa0_lsin = df2['Lsin'].mean()
pcsa0_lsin_sem = df2['Lsin'].sem(ddof=0)
avgpcsa0_lcos = df2['Lcos'].mean()
pcsa0_lcos_sem = df2['Lcos'].sem(ddof=0)
print('Average of',len(df2['Avg Length']), 'Trials :', avgpcsa0_length, '+/-',
      pcsa0_sem)
#print('Average cathodal :', avgpcsa0_lcathode, '+/-', pcsa0_lcathode_sem)
#print('Average anodal :', avgpcsa0_lanode, '+/-', pcsa0_lanode_sem)
print('===T-Tests===')
from scipy import stats
lenstats = stats.ttest_ind(df['Avg Length'], df2['Avg Length'], equal_var=True)
biasstats_sin = stats.ttest_ind(df['Lsin'], df2['Lsin'], equal_var=True)
biasstats_cos = stats.ttest_ind(df['Lcos'], df2['Lcos'], equal_var=True)
print(lenstats)
print(biasstats_sin)
print(biasstats_cos)
```

## Plotting

The code below produces plots with error bars (standard error of the mean) and significance  (p values).

```python
#plotting
import matplotlib.pyplot as plt
n_groups = 1
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.5
opacity = 0.7
error_config = {'ecolor': '0.1', 'capsize' : 3}
data = (avgpcsa15_length, avgpcsa0_length, avgpcsa15_length, avgpcsa0_length)
rects1 = plt.bar(index, avgpcsa15_length, bar_width,
                 alpha=opacity,
                 color='r',
                 yerr=pcsa15_sem,
                 error_kw=error_config,
                 label='15V PCSA')
rects2 = plt.bar(index + 1.25*bar_width, avgpcsa0_length, bar_width,
                 alpha=opacity,
                 color='b',
                 yerr=pcsa0_sem,
                 error_kw=error_config,
                 label='0V PCSA')

plt.annotate("", xy=(0.0, 62.6), xycoords='data',
           xytext=(0.625, 62.6), textcoords='data',
           arrowprops=dict(arrowstyle="-", ec='black',
                           connectionstyle="bar,fraction=1.09", 
                           shrinkA=0, shrinkB=66))
plt.text(0.30, 70.7, 'p<0.05',
       horizontalalignment='center',
       verticalalignment='center')

plt.axis([-1.25,2,50,75])
plt.xlabel('')
plt.ylabel(u'Average Neurite Length (${\mu}m$)', fontsize = 12)
plt.title('')
plt.xticks([])
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('length.png', dpi=300)
plt.close()


fig, (ax1, ax2) = plt.subplots(figsize=(6.2,4), ncols=2)
index = np.arange(1)
bar_width = 0.5
opacity = 0.7
error_config = {'ecolor': '0.1', 'capsize' : 3}

ax1.barh(index, avgpcsa15_lcos, bar_width,
        alpha=opacity,
        color='r',
        xerr=pcsa15_lcos_sem,
        error_kw=error_config,
        label='15V PCSA')
ax1.barh(index + 1.25*bar_width, avgpcsa0_lcos, bar_width,
        alpha=opacity,
        color='b',
        xerr=pcsa0_lcos_sem,
        error_kw=error_config,
        label='0V PCSA')
ax1.set_yticks([])
ax1.set_xticks([-0.5,0,0.5,1,1.5])

ax2.bar(index, avgpcsa15_lsin, bar_width,
        alpha=opacity,
        color='r',
        yerr=avgpcsa0_lsin,
        error_kw=error_config,
        label='15V PCSA')
ax2.bar(index + 1.25*bar_width, avgpcsa0_lsin, bar_width,
        alpha=opacity,
        color='b',
        yerr=pcsa0_lsin_sem,
        error_kw=error_config,
        label='0V PCSA')
ax2.axis([-0.3,0.9,-11.5,1])
ax2.set_xticks([0.3])
ax2.tick_params(axis='x', colors='white')

ax1.axvline(0, color='black', lw=1)
ax2.axhline(0, color='black',lw=1)

plt.annotate("", xy=(0.625, -2.3), xycoords='data',
           xytext=(0, -2.3), textcoords='data',
           arrowprops=dict(arrowstyle="-", ec='black',
                           connectionstyle="bar,fraction=1.73", 
                           shrinkA=132, shrinkB=0))
ax2.text(0.3, -10.85, 'p<0.05',
       horizontalalignment='center',
       verticalalignment='center')

bbox_props = dict(boxstyle="rarrow,pad=0.25", fc="white", ec="black", lw=2)
t = ax2.text(1.08, -5.2, "    Electric Field    ", ha="center", va="center", 
            rotation=-90, size=15, bbox=bbox_props)
bbox_props = dict(boxstyle="rarrow,pad=0.25", fc="white", ec="black", lw=2)
t = ax1.text(-0.9, 0.32, "    Electric Field    ", ha="center", va="center", 
            rotation=-90, size=15, bbox=bbox_props)

ax1.set_xlabel('Perpendicular Bias')
ax2.set_xlabel('Parallel Bias')
plt.text(-1.4, 1.45, 'Growth Dependent Response to E-Field', dict(size=12))

fig.legend(bbox_to_anchor=[0.94, 1.03])
plt.show()
fig.savefig('bias.png', bbox_inches = 'tight', dpi=300)
plt.close('all')
```

![Figure 4. Average Neurite Length](https://i.imgur.com/rHP5rrB.png)

![Figure 5. Bias or Directional Preference](https://i.imgur.com/LcJ77N7.png)

## Conclusion

Electric field stimulation resulted in PC12 growing longer neurites overall (Figure 4). More importantly, I have shown that neurites growing toward the anode  of the device (parallel to the electric field) grew longer than those growing perpendicularly (at 90 degree angles) to the field (Figure 5).
