import math
import os
import numpy as np
nlength = list() #includes avg of single trial
nangle = list() #includes avg of single trial
vectcos = list()
vectsin = list()
radians = list()
l_cathode = list()
l_anode = list()
pcsa15_length = list() #includes avgs of trials
pcsa15_lcos = list() #includes avgs of trials
pcsa15_lsin = list() #includes avgs of trials
pcsa0_length = list() #includes avgs of trials
pcsa0_lcos = list() #includes avgs of trials
pcsa0_lsin = list() #includes avgs of trials
avg15_lcathode = list() #includes avgs of trials
avg15_lanode = list() #includes avgs of trials
avg0_lcathode = list() #includes avgs of trials
avg0_lanode = list() #includes avgs of trials

def neurimport() :
    count = 0
    templen = list()
    tempang = list()
    temprad = list()
    cosang = list()
    sinang = list()
    #extract data
    for line in handle :
        if line.startswith(' ') :
            continue
        line = line.rstrip()
        line = line.split('\t')
        length = float(line[4])
        templen.append(length)
        angle = float(line[3])
        tempang.append(angle)
        count = count + 1
    #correction in case angles were measured first
    counter = 0
    for i in tempang[0 : int(count/2)] :
        if not i == 0 :
            counter = counter + 1
        else :
            continue
    if int(counter) > 0 :
        #correction delete
        del templen[0 : int(count/2)]
        del tempang[int(count/2) : ]
    if int(counter) == 0 :
        #delete unwanted data
        del templen[int(count/2) : ]
        del tempang[0 : int(count/2)]
    #convert negative angles
    for ang in tempang :
        if ang < 0 :
            tempang[tempang.index(ang)] = ang + 360
    #conversion to radians
    for i in tempang :
        ang = float(i) * (2 * math.pi)/360
        temprad.append(ang)
    #compute cos and sin of angles
    for i in temprad :
        cos = math.cos(float(i))
        sin = math.sin(float(i))
        cosang.append(cos)
        sinang.append(sin)
    #compute vector components
    for i in range(0 , int(len(templen))) :
        vcos = templen[i] * cosang[i]
        vsin = templen[i] * sinang[i]
        vectcos.append(vcos)
        vectsin.append(vsin)
    for i in templen :
        nlength.append(i)
    for i in tempang :
        nangle.append(i)
    for i in temprad :
        radians.append(i)
    #break lengths into anode and cathode
    cnt = 0
    for i in nangle :
        if i >= 0 and i < 180 :
            l_cathode.append(nlength[cnt])
        else :
            l_anode.append(nlength[cnt])
        cnt = cnt + 1
        
#Trial 1
path = 'Neurite_data/15V PCSA/1/'
listing = os.listdir(path)
for file in listing :
    handle = open(os.path.join(path,file) , 'r')
    a = neurimport()
    #compute averages
    avglen = sum(nlength) / len(nlength) #avg length for single trial
    avgvcos = sum(vectcos) / len(vectcos) #perp bias for single trial
    avgvsin = sum(vectsin) / len(vectsin) #parallel bias for single trial
    lcathode = sum(l_cathode) / len(l_cathode)
    lanode = sum(l_anode) / len(l_anode)

pcsa15_length.append(avglen)
pcsa15_lcos.append(avgvcos)
pcsa15_lsin.append(avgvsin)
avg15_lcathode.append(lcathode)
avg15_lanode.append(lanode)

#Display data
#print('Lengths', nlength)
#print('Angles', nangle)
print('Avg Length', avglen)
print('L*cos(theta)', avgvcos)
print('L*sin(theta)', avgvsin)

del nlength[:]
del nangle[:]
del vectcos[:]
del vectsin[:]
del radians[:]
del l_cathode[:]
del l_anode[:]

#Trial 2
path = 'Neurite_data/15V PCSA/2/'
listing = os.listdir(path)
for file in listing :
    handle = open(os.path.join(path,file) , 'r')
    a = neurimport()
    #compute averages
    avglen = sum(nlength) / len(nlength)
    avgvcos = sum(vectcos) / len(vectcos)
    avgvsin = sum(vectsin) / len(vectsin)
    lcathode = sum(l_cathode) / len(l_cathode)
    lanode = sum(l_anode) / len(l_anode)

pcsa15_length.append(avglen)
pcsa15_lcos.append(avgvcos)
pcsa15_lsin.append(avgvsin)
avg15_lcathode.append(lcathode)
avg15_lanode.append(lanode)

#Display data
#print('Lengths', nlength)
#print('Angles', nangle)
print('Avg Length', avglen)   
print('L*cos(theta)', avgvcos)
print('L*sin(theta)', avgvsin)

del nlength[:]
del nangle[:]
del vectcos[:]
del vectsin[:]
del radians[:]
del l_cathode[:]
del l_anode[:]

#Trial3
path = 'Neurite_data/15V PCSA/3/'
listing = os.listdir(path)
for file in listing :
    handle = open(os.path.join(path,file) , 'r')
    a = neurimport()
    #compute averages
    avglen = sum(nlength) / len(nlength)
    avgvcos = sum(vectcos) / len(vectcos)
    avgvsin = sum(vectsin) / len(vectsin)
    lcathode = sum(l_cathode) / len(l_cathode)
    lanode = sum(l_anode) / len(l_anode)

pcsa15_length.append(avglen)
pcsa15_lcos.append(avgvcos)
pcsa15_lsin.append(avgvsin)
avg15_lcathode.append(lcathode)
avg15_lanode.append(lanode)

#Display data
#print('Lengths', nlength)
#print('Angles', nangle)
print('Avg Length', avglen)   
print('L*cos(theta)', avgvcos)
print('L*sin(theta)', avgvsin)

del nlength[:]
del nangle[:]
del vectcos[:]
del vectsin[:]
del radians[:]
del l_cathode[:]
del l_anode[:]

#Trial 1
path = 'Neurite_data/0V PCSA/1/'
listing = os.listdir(path)
for file in listing :
    handle = open(os.path.join(path,file) , 'r')
    a = neurimport()
    #compute averages
    avglen = sum(nlength) / len(nlength)
    avgvcos = sum(vectcos) / len(vectcos)
    avgvsin = sum(vectsin) / len(vectsin)
    lcathode = sum(l_cathode) / len(l_cathode)
    lanode = sum(l_anode) / len(l_anode)

pcsa0_length.append(avglen)
pcsa0_lcos.append(avgvcos)
pcsa0_lsin.append(avgvsin)
avg0_lcathode.append(lcathode)
avg0_lanode.append(lanode)

#Display data
#print('Lengths', nlength)
#print('Angles', nangle)
print('Avg Length', avglen)   
print('L*cos(theta)', avgvcos)
print('L*sin(theta)', avgvsin)

del nlength[:]
del nangle[:]
del vectcos[:]
del vectsin[:]
del radians[:]
del l_cathode[:]
del l_anode[:]

#Trial 2
path = 'Neurite_data/0V PCSA/2/'
listing = os.listdir(path)
for file in listing :
    handle = open(os.path.join(path,file) , 'r')
    a = neurimport()
    #compute averages
    avglen = sum(nlength) / len(nlength)
    avgvcos = sum(vectcos) / len(vectcos)
    avgvsin = sum(vectsin) / len(vectsin)
    lcathode = sum(l_cathode) / len(l_cathode)
    lanode = sum(l_anode) / len(l_anode)

pcsa0_length.append(avglen)
pcsa0_lcos.append(avgvcos)
pcsa0_lsin.append(avgvsin)
avg0_lcathode.append(lcathode)
avg0_lanode.append(lanode)

#Display data
#print('Lengths', nlength)
#print('Angles', nangle)
print('Avg Length', avglen)   
print('L*cos(theta)', avgvcos)
print('L*sin(theta)', avgvsin)

del nlength[:]
del nangle[:]
del vectcos[:]
del vectsin[:]
del radians[:]
del l_cathode[:]
del l_anode[:]

#Trial3
path = 'Neurite_data/0V PCSA/3/'
listing = os.listdir(path)
for file in listing :
    handle = open(os.path.join(path,file) , 'r')
    a = neurimport()
    #compute averages
    avglen = sum(nlength) / len(nlength)
    avgvcos = sum(vectcos) / len(vectcos)
    avgvsin = sum(vectsin) / len(vectsin)
    lcathode = sum(l_cathode) / len(l_cathode)
    lanode = sum(l_anode) / len(l_anode)

pcsa0_length.append(avglen)
pcsa0_lcos.append(avgvcos)
pcsa0_lsin.append(avgvsin)
avg0_lcathode.append(lcathode)
avg0_lanode.append(lanode)

#Display data
#print('Lengths', nlength)
#print('Angles', nangle)
print('Avg Length', avglen)   
print('L*cos(theta)', avgvcos)
print('L*sin(theta)', avgvsin)

del nlength[:]
del nangle[:]
del vectcos[:]
del vectsin[:]
del radians[:]
del l_cathode[:]
del l_anode[:]

#Statistics
print('===15V PCSA===')
avgpcsa15_length = float(sum(pcsa15_length)/len(pcsa15_length))
pcsa15_sem = np.std(pcsa15_length)/math.sqrt(len(pcsa15_length))
avgpcsa15_lcathode = float(sum(avg15_lcathode)/len(avg15_lcathode))
pcsa15_lcathode_sem = np.std(avg15_lcathode)/math.sqrt(len(avg15_lcathode))
avgpcsa15_lanode = float(sum(avg15_lanode)/len(avg15_lanode))
pcsa15_lanode_sem = np.std(avg15_lanode)/math.sqrt(len(avg15_lanode))
avgpcsa15_lsin = float(sum(pcsa15_lsin)/len(pcsa15_lsin))
pcsa15_lsin_sem = np.std(pcsa15_lsin)/math.sqrt(len(pcsa15_lsin))
avgpcsa15_lcos = float(sum(pcsa15_lcos)/len(pcsa15_lcos))
pcsa15_lcos_sem = np.std(pcsa15_lcos)/math.sqrt(len(pcsa15_lcos))
print('Average of',len(pcsa15_length), 'Trials :', avgpcsa15_length, '+/-',
      pcsa15_sem)
print('Average cathodal :', avgpcsa15_lcathode, '+/-', pcsa15_lcathode_sem)
print('Average anodal :', avgpcsa15_lanode, '+/-', pcsa15_lanode_sem)

print('===0V PCSA===')
avgpcsa0_length = float(sum(pcsa0_length)/len(pcsa0_length))
pcsa0_sem = np.std(pcsa0_length)/math.sqrt(len(pcsa0_length))
avgpcsa0_lcathode = float(sum(avg0_lcathode)/len(avg0_lcathode))
pcsa0_lcathode_sem = np.std(avg0_lcathode)/math.sqrt(len(avg0_lcathode))
avgpcsa0_lanode = float(sum(avg0_lanode)/len(avg0_lanode))
pcsa0_lanode_sem = np.std(avg0_lanode)/math.sqrt(len(avg0_lanode))
avgpcsa0_lsin = float(sum(pcsa0_lsin)/len(pcsa0_lsin))
pcsa0_lsin_sem = np.std(pcsa0_lsin)/math.sqrt(len(pcsa15_lsin))
avgpcsa0_lcos = float(sum(pcsa0_lcos)/len(pcsa0_lcos))
pcsa0_lcos_sem = np.std(pcsa0_lcos)/math.sqrt(len(pcsa0_lcos))
print('Average of',len(pcsa0_length), 'Trials :', avgpcsa0_length, '+/-',
      pcsa0_sem)
print('Average cathodal :', avgpcsa0_lcathode, '+/-', pcsa0_lcathode_sem)
print('Average anodal :', avgpcsa0_lanode, '+/-', pcsa0_lanode_sem)
print('===T-Tests===')
from scipy import stats
lenstats = stats.ttest_ind(pcsa15_length, pcsa0_length, equal_var=True)
biasstats_sin = stats.ttest_ind(pcsa15_lsin, pcsa0_lsin, equal_var=True)
biasstats_cos = stats.ttest_ind(pcsa15_lcos, pcsa0_lcos, equal_var=True)
print(lenstats)
print(biasstats_sin)
print(biasstats_cos)


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