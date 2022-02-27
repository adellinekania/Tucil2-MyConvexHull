# MAIN PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from myConvexHull import myConvexHull

# Memberi tahu pengguna dataset yang tersedia
print('Available Datasets: ')

availableDatasets = ['Irish', 'Breast Cancer', 'Wine']
maxRowDatasets = [3, 29, 12]

for i in range(len(availableDatasets)):
    print(i+1, '. ', availableDatasets[i])

# Meminta input nomor dataset kepada pengguna
# Asumsi input selalu valid
inputDataset = int(input('Select dataset (1-3): '))

# Meload dataset
if inputDataset == 1:
    data = datasets.load_iris()
elif inputDataset == 2:
    data = datasets.load_breast_cancer()
else:
    data = datasets.load_wine()

# Membuat dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = pd.DataFrame(data.target)

# Meminta input berupa index kolom yang akan dijadikan axis dan ordinat
# Asumsi input selalu berada didalam range yang diberikan
x = int(input(f'Enter column index for axis (0-{maxRowDatasets[inputDataset-1]}): '))
y = int(input(f'Enter column index for ordinate (0-{maxRowDatasets[inputDataset-1]}): '))

# Visualisasi hasil Convex Hull
plt.figure(figsize = (10, 6))
colors = ['b','r','g']

xName = str.title(data.feature_names[x])
yName = str.title(data.feature_names[y])
title = xName + ' vs ' + yName
plt.title(title)

plt.xlabel(data.feature_names[x])
plt.ylabel(data.feature_names[y])

for i in range(len(data.target_names)):
    bucket = df[df['Target'] == i].iloc[:,[x,y]].values
    myData = df[df['Target'] == i].iloc[:,[x,y]].values
    # Menggunakan fungsi myConvexHull untuk menentukan titik-titik
    # yang membentuk Convex Hull dari dataset yang diberikan
    hull = myConvexHull(myData)
    plt.scatter(bucket[:, 0], bucket[:, 1], label=data.target_names[i])
    for simplex in hull:
        plt.plot(bucket[simplex, 0], bucket[simplex, 1], colors[i])
        
plt.show()