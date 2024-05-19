from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv
from sklearn.cluster import KMeans#For elbow method only
from scipy.spatial.distance import cdist
   



datafile=open('iris.data')
In_text = csv.reader(datafile,delimiter = ',')
 
csvfile =open('iris.csv','w')
out_csv = csv.writer(csvfile)
out_csv.writerow(['SepalLength','SepalWidth','PetalLength','PetalWidth','FClass']) 
file3 = out_csv.writerows(In_text)
 
datafile.close()
csvfile.close()
Idata = pd.read_csv("iris.csv") #load the dataset
Idata.head()

# Change categorical data to number 0-2
Idata["FClass"] = pd.Categorical(Idata["FClass"])
Idata["FClass"] = Idata["FClass"].cat.codes
# Change dataframe to numpy matrix
data = Idata.values[:, 0:4]
category = Idata.values[:, 4]


#determining the best k value

dis = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data)
    kmeanModel.fit(data)
    dis.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

plt.plot(K, dis, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

k = 3 # getting the elbow value from the curve and hard coding it


n = data.shape[0]# assigning training data shape
c = data.shape[1] # assigning features data shape



# Firstly generating random centroids using variance and mean

mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
center = np.random.randn(k,c)*std + mean

prev_cent = np.zeros(center.shape) # to store old center
new_cent = deepcopy(center) # Store new center

data.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(new_cent - prev_cent)



# Generating best centroids and the 3 clusters to plot them

while error != 0:
    for i in range(k):
        distances[:,i] = np.linalg.norm(data - center[i], axis=1)
    clusters = np.argmin(distances, axis = 1)    
    prev_cent = deepcopy(new_cent)
    for i in range(k):
        new_cent[i] = np.mean(data[clusters == i], axis=0)
    error = np.linalg.norm(new_cent - prev_cent)
incorrect = 0


colors=['red', 'blue', 'green']
for i in range(n):
    if colors[int(category[i])] == 'red':
        l1 = plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])], label = 'Iris-setosa')
    elif colors[int(category[i])] == 'blue':
        l2 = plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])], label = 'Iris-versicolor')
    elif colors[int(category[i])] == 'green':
        l3 = plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])], label = 'Iris-virginica')
handles, labels = plt.gca().get_legend_handles_labels()
handle_list, label_list = [], []

for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)

plt.legend(handle_list, label_list)
plt.scatter(new_cent[:,0], new_cent[:,1], marker='^', c='black', s=150, label = 'Centroids')
plt.show()

print("The following are the 3 centroids of the k=3 means clusetering:")
print(new_cent)