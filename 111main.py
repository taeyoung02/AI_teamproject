import numpy as np
import readData as rd
import DNN


data_path = "AI/project/data"
train_data = rd.get_input(data_path + "/X_train.csv")
target_data = rd.get_result(data_path + "/y_train.csv")

test_data = rd.get_input(data_path + "/X_test.csv")
result_data = rd.get_result(data_path + "/y_test.csv")

print(train_data.shape)

DNN.DNN1(train_data, target_data, test_data, result_data)
DNN.DNN2(train_data, target_data, test_data, result_data)
DNN.DNN3(train_data, target_data, test_data, result_data)
DNN.DNN4(train_data, target_data, test_data, result_data)
DNN.DNN5(train_data, target_data, test_data, result_data)
DNN.DNN6(train_data, target_data, test_data, result_data)
DNN.DNN7(train_data, target_data, test_data, result_data)
DNN.DNN8(train_data, target_data, test_data, result_data)
DNN.DNN9(train_data, target_data, test_data, result_data)
DNN.DNN10(train_data, target_data, test_data, result_data)
DNN.DNN11(train_data, target_data, test_data, result_data)







