import glob
import numpy as np
import pylab as plt

def get_timing(f):
    t = {}
    with open(f,'r') as f:
        for line in f.readlines():
            k = line.split(',')[0]
            values = line.split(',')[1:]
            value = max([float(v)/3600. for v in values])
            t[k] = value
    return t

timing_files = glob.glob("L*/timing.txt")

h = {}
labels = []
num_keys = {}
total = []
for f in timing_files:
    t = get_timing(f)
    for k in t.keys():
        if k not in labels:
            labels.append(k)
        num_keys[f] = list(t.keys())

has_all_keys = []
for f in timing_files:
    has_all_keys.append(np.all(np.isin(labels, num_keys[f])))

for i, f in enumerate(timing_files):
    t = get_timing(f) 
    
    total.append(np.sum(list(t.values())))
    if not has_all_keys[i]:
        continue
    #print(f,t)
    for k in t.keys():
        if k not in h.keys():
            h[k] = [t[k]]
        else:
            h[k].append(t[k])

total=np.array(total)
total_done = total[has_all_keys]
for k in h.keys():
    print("{:<24} -> median {:.1f} | mean {:.1f} hours | std {:.1f} hours | max {:.1f} hours | min {:.1f}".format(k, np.median(h[k]), np.mean(h[k]), np.std(h[k]), np.max(h[k]), np.min(h[k])))

print("Total -> median {:.1f} mean {:.1f} hours | std {:.1f} hours | max {:.1f} hours | min {:.1f}".format(np.median(total_done), np.mean(total_done), np.std(total_done), np.max(total_done), np.min(total_done)))

argsort = np.argsort(total)[::-1]
for i, j in enumerate(argsort):
    print("{}{} : {:<24} took {:.1f} hours".format(i, "*" if not has_all_keys[j] else " ", timing_files[j], total[j]))

