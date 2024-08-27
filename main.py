# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
from itertools import combinations_with_replacement as cwr




def length_of_design_matrix(m, d):
    return math.factorial(m+d)/(math.factorial(d)*math.factorial(m))


def design_matrix(m, x,vec):
    vec.clear()
    for i in range(m+1):
        z = cwr(x, i)
        products = [np.prod(comb) for comb in z]
        vec = vec + products
    return vec


def error_rms(predicted, expected):
    x = [(predicted[i]-expected[i])**2 for i in range(len(predicted))]
    return np.mean(x)**0.5

def predict(m,weights,x):
    z=design_matrix(m,x,[])
    return np.dot(np.transpose(weights),z)
#(3x**3-100x**2+72*x)/1000 + 50

def targ(x):
    return math.sin(x)
def model(total_data,target,m):
    D = int(length_of_design_matrix(m, Dimensions));
    train_data = total_data[:int(len(total_data)*0.7),:]
    train_target = target[:int(len(total_data)*0.7),:]
    # valid_data = total_data[int(len(total_data)*0.7):int(len(total_data)*0.9)]
    # valid_target = target[int(len(total_data)*0.7):int(len(total_data)*0.9)]
    test_data = total_data[int(len(total_data)*0.7):,:]
    test_target = target[int(len(total_data)*0.7):,:]
    basis_func = np.zeros((len(train_data),D))
    for i in range(len(train_data)):
        basis_func[i] = design_matrix(m,train_data[i],[])
    # print(basis_func)
    # print(basis_func.shape)
    weights = np.linalg.pinv(basis_func) @ train_target
    print(weights.shape)
    predicted_vals_test = [predict(m,weights, x) for x in test_data]
    predicted_vals_test = np.array(predicted_vals_test)
    predicted_vals_train = [predict(m, weights, x) for x in train_data]
    predicted_vals_train = np.array(predicted_vals_train)
    return [error_rms(predicted_vals_test,test_target),error_rms(predicted_vals_train,train_target),weights]
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_size = 100
    m = 15
    Dimensions = 1
    D = int(length_of_design_matrix(m,Dimensions));
    total_data = np.zeros((data_size,Dimensions))
    target = np.zeros((data_size,1))
    for i in range(data_size):
        total_data[i][0] = (random.uniform(-10,10))
        target[i][0] = (targ(total_data[i][0]))
    # target = target + np.random.uniform(-5,5,(data_size,1))

    # data = {'input': total_data,'output': target}
    # file = pd.DataFrame(data)
    # file.to_csv('Data.csv', index=False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
    Erms_train = []
    Erms_test = []
    for m in range(1,25):
        x = model(total_data,target,m)
        print(x[2])
        Erms_test.append(x[0])
        Erms_train.append(x[1])

    model_orders = list(range(1, 25))

# Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(model_orders, Erms_train, marker='o', linestyle='-', color='b', label='Erms Train')
    plt.plot(model_orders, Erms_test, marker='o', linestyle='--', color='r', label='Erms Test')

# Add labels and title
    plt.xlabel('Model Order M')
    plt.ylabel('Erms (Root Mean Square Error)')
    plt.title('Training and Testing Erms vs. Model Order')
    plt.legend()
    plt.grid(True)

# Show the plot
    plt.show()
