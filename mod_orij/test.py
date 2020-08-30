
import RF_fextract
import dill
import random
import os
from collections import defaultdict
import argparse
from itertools import chain
import pickle
import numpy as np
from sklearn.metrics import accuracy_score 
hs_test_data = r"/home/muskan/Desktop/newexp/HS_test/"
dic_of_feature_data = r"/home/muskan/Desktop/newexp/mod_orij/hs_test/"

hs_sites = 30
hs_instances = 20

def create_dict(path_to_dict = dic_of_feature_data, path_to_hs = hs_test_data, hs_sites = 30, hs_instances = 20):

    data_dict = {'hs_feature': [], 
                 'hs_label': [] }
                 
    print("Creating HS features...")
    for i in range(1, hs_sites + 1):
        for j in range(80,100):
            fname = str(i) + "_" + str(j) + ".txt"
            if os.path.exists(path_to_hs + fname):
                tcp_dump = open(path_to_hs + fname).readlines()
                g = []
                g.append(RF_fextract.TOTAL_FEATURES(tcp_dump))
                data_dict['hs_feature'].append(g)
                data_dict['hs_label'].append((i,j))
        print(i)
    
    print("HS Dictionary created")
    assert len(data_dict['hs_feature']) == len(data_dict['hs_label'])
    
    pkl_test = "test_features.pkl"
    with open(pkl_test, 'wb') as file:
         pickle.dump(data_dict, file)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def testing_hs(path_to_dict = dic_of_feature_data):
    pkl_test = "test_features.pkl"
    with open (pkl_test, 'rb') as fileObject1:
         dic = pickle.load(fileObject1)

    split_data = list(chunks(dic['hs_feature'], hs_instances))
    split_target = list(chunks(dic['hs_label'], hs_instances))

    test_data = []
    test_label = []
    for i in range(len(split_data)):
        temp = list(zip(split_data[i], split_target[i]))
        random.shuffle(temp)
        data,label = zip(*temp)
        test_data.extend(data)
        test_label.extend(label)
     

    flat_test_data = []
    for te in test_data:
        flat_test_data.append(list(sum(te, ())))
    test_features =  list(zip(flat_test_data, test_label))
    return test_features

def test_data(path_to_dict = dic_of_feature_data):
    test = testing_hs(path_to_dict)
    te_data, te_label1 = zip(*test)
    #xtest = 
    te_label = list(zip(*te_label1))[0]
    #print(te_data)
    y_test = np.asarray(te_label)
    #print(y_test)
    print("Testing ...")
    
    myfile = "rfmodel.dill"
    with open(myfile, 'rb') as file:
         dill_model = dill.load(file)
         ypred = dill_model.predict(te_data)
         #print(ypred)
         #print("hiiii")
         
         acc = accuracy_score(y_test, ypred)
         print("Accuracy = %.2f"%(100*acc))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='k-FP benchmarks')
    parser.add_argument('--dictionary', action='store_true', help='Build dictionary.')
    parser.add_argument('--test', action='store_true', help='Closed world classification.')
    args = parser.parse_args()

    if args.dictionary:

        create_dict()

    elif args.test:

        test_data()

     
           


