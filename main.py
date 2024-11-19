import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y_test = test_data['Y'].iloc[:1000].tolist()

train = train_data.iloc[:6000].to_numpy()
test = test_data.iloc[:1000].to_numpy()

y_test1 = train_data['Y'].iloc[:1000].tolist()

test1 = train_data.iloc[:1000].to_numpy()



def euclidean_distance(sample1, sample2):
    return np.linalg.norm(sample1[:-1] - sample2[:-1])



def voting(array):
    second_elements = [row[1] for row in array]
    vote_counts = Counter(second_elements)
    vote = vote_counts.most_common(1)[0][0]
    return vote

def KNN_kernel(train_d,test_d,K):
	dist = []
	dist = [[euclidean_distance(train_d[i], test_d), train_d[i][-1]] for i in range(len(train_d))]
	soretd_dist = sorted(dist, key=lambda x: x[0])
	return voting(soretd_dist[:K])


K = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
error_list = []

# for k in K:
# 	y_pred = []
# 	for t in test:
# 		y_pred.append(KNN_kernel(train,t,k))

# 	accuracy = accuracy_score(y_test, y_pred)
# 	error = 1 - accuracy

# 	print(error)
	
# 	error_list.append([k,error])
error_list1 = []
for k in K:
	y_pred = []
	for t in test1:
		y_pred.append(KNN_kernel(train,t,k))

	accuracy = accuracy_score(y_test1, y_pred)
	error = 1 - accuracy

	print(error)
	
	error_list1.append([k,error])


error_list = [[1, 0.014000000000000012], [9, 0.015000000000000013], [19, 0.028000000000000025], [29, 0.03500000000000003], [39, 0.040000000000000036], [49, 0.040000000000000036], [59, 0.04600000000000004], [69, 0.049000000000000044], [79, 0.051000000000000045], [89, 0.05400000000000005], [99, 0.05300000000000005]]
x_values = [point[0] for point in error_list]
y_values = [point[1] for point in error_list]

x_values1 = [point[0] for point in error_list1]
y_values1 = [point[1] for point in error_list1]


plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
plt.plot(x_values1, y_values1, marker='o', linestyle='-', color='r')
plt.xlabel("K value")
plt.ylabel("Error Rate")

plt.grid()


plt.savefig("plot.png", dpi=300)  
plt.show()