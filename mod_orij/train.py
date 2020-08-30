import csv
import sys
from sys import stdout
import RF_fextract
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import scipy
import dill
import random
import os
from collections import defaultdict
import argparse
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib import pyplot
import time 

hs_training_data = r"/home/muskan/Desktop/newexp/HS_train/"
dic_of_feature_data = r"/home/muskan/Desktop/newexp/mod_orij/hs_features"

hs_sites = 30
hs_instances = 80

def create_dict(path_to_dict = dic_of_feature_data, path_to_hs = hs_training_data, hs_sites = hs_sites, hs_instances = hs_instances):

    dic_of_feature_data = path_to_dict
    data_dict = {'hs_feature': [], 
                 'hs_label': [] }
                 
    print("Creating HS features...")
    for i in range(1, hs_sites + 1):
        for j in range(hs_instances):
            fname = str(i) + "_" + str(j) + ".txt"
            if os.path.exists(path_to_hs + fname):
                tcp_dump = open(path_to_hs + fname).readlines()
                #print(tcp_dump)
                g = []
                g.append(RF_fextract.TOTAL_FEATURES(tcp_dump))
                data_dict['hs_feature'].append(g)
                data_dict['hs_label'].append((i,j))
                #print(data_dict)
        print(i)

    print("HS Dictionary created")
    assert len(data_dict['hs_feature']) == len(data_dict['hs_label'])
    fileObject = open(dic_of_feature_data,'wb')
    dill.dump(data_dict,fileObject)
    fileObject.close()

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]
def checkequal(lst):
    return lst[1:] == lst[:-1]


def training_hs(path_to_dict = dic_of_feature_data):

    fileObject1 = open(path_to_dict,'rb')
    dic = dill.load(fileObject1)

    split_data = list(chunks(dic['hs_feature'], hs_instances))
    split_target = list(chunks(dic['hs_label'], hs_instances))

    training_data = []
    training_label = []
    for i in range(len(split_data)):
        temp = list(zip(split_data[i], split_target[i]))
        random.shuffle(temp)
        data, label = zip(*temp)
        training_data.extend(data)
        training_label.extend(label)
        
    flat_train_data = []
    for tr in training_data:
        flat_train_data.append(list(sum(tr, ())))
    training_features =  list(zip(flat_train_data, training_label))
    return training_features

        
    flat_train_data = []
    for tr in training_data:
        flat_train_data.append(list(sum(tr, ())))
    training_features =  list(zip(flat_train_data, training_label))
    return training_features

def train_data(path_to_dict = dic_of_feature_data):
    
    training = training_hs(path_to_dict)
    tr_data, tr_label1 = zip(*training)
    tr_label = list(zip(*tr_label1))[0]
    print("Training ...")
    trdata = np.asarray(tr_data)

    start = time.time()
    model = RandomForestClassifier(n_jobs=-1, n_estimators=1000, oob_score=True)
    model.fit(trdata, tr_label)
    end = time.time()
    shape = np.shape(tr_data)
    print("Execution time for building the Tree is: %f"%(float(end)- float(start)))

    print("Size of Data set before feature selection: %.2f MB"%(trdata.nbytes/1e6))
    print("Shape of the dataset ",shape)
    ##print(model.feature_importances_)
    #print(m)
    #pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    #pyplot.show()

    myfile = 'rfmodel.dill'
    with open (myfile, 'wb') as file:
         dill.dump(model,file)
         print(model)
                       
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='k-FP benchmarks')
    parser.add_argument('--dictionary', action='store_true', help='Build dictionary.')
    parser.add_argument('--train', action='store_true', help='Closed world classification.')
    args = parser.parse_args()

    if args.dictionary:

        create_dict()

    elif args.train:

        train_data()
     

