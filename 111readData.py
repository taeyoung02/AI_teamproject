import pandas as pd
import numpy as np

def get_input(path):
    exfile = pd.read_csv(path)
    exlist = exfile.values.tolist()
    exlist = np.delete(exlist,0,axis=1)
    exlist = np.delete(exlist,0,axis=1)
    return exlist

def get_result(path):
    exfile = pd.read_csv(path)
    exlist = exfile.values.tolist()
    exlist = np.delete(exlist,0,axis=1)
    return exlist

'''
path = "AI/project/data/X_test.csv"
path2 = "AI/project/data/y_test.csv"
aa = get_input(path)
bb = get_result(path2)
#a1 = np.delete(aa,0,axis=1)
#b1 = np.delete(bb,0,axis=1)
print(aa)
print(bb)
'''
