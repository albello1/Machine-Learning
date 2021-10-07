# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:06:25 2021

@author: Albert
"""

"""
Created on Wed Mar 10 21:54:30 2021

@author: Albert
"""

import numpy as np
from cargar_ima_PET_mask import cargar_ima_PET_mask
import pandas as pd


from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import matplotlib.pyplot as plt
import multiprocessing as mp
from random import randrange
from sklearn.gaussian_process import GaussianProcessClassifier as GP
import GPy

def main3(data_path_ima1, mask_path_ima1, data_path_ima2, mask_path_ima2):
    
    
    data1, data_pos1, labels1 = cargar_ima_PET_mask(data_path = data_path_ima1, mask_path = mask_path_ima1)
    data2, data_pos2, labels2 = cargar_ima_PET_mask(data_path = data_path_ima2, mask_path = mask_path_ima2)
#    acc_train = []
#    acc_test = []
#    nums = np.zeros((30,3))

#        a = randrange(1)
#        x = randrange(0,a)
#        b = randrange(a,1)
#        y = b -x
#        z = 1 - y
    cut = np.shape(data1[1])
    data = np.concatenate((data1,data2),axis = 1)
    labels = np.concatenate((labels1,labels2),axis = 0)
    #Vamos a crear un kernel lineal para la formulacion dual con un SVC
    #Primero preparamos el cross validation raro
    train_acc = np.zeros((data.shape[0]))
    test_acc = np.zeros((data.shape[0]))
    params = np.zeros((data.shape[0],data.shape[1]))
    ard_coef = np.zeros((data1.shape[0],data1.shape[1]))
    for i in range(data1.shape[0]):
                
        labels_new = labels1
        test_set1 = data[i,:]
        test_set1 = np.reshape(test_set1, (1,test_set1.shape[0]))
        Y_test = labels_new[i]
        Y_test = np.array(Y_test)
        train_set1 = [x for j,x in enumerate(data) if j!=i]
        train_set1 = np.array(train_set1)
        Y_train =  [x for j,x in enumerate(labels) if j!=i] 
        Y_train = np.reshape(Y_train, (len(Y_train),1))
                
                
        kernel_train = GPy.kern.Linear(train_set1.shape[1], ARD = True) 
        kernel_test = GPy.kern.Linear(test_set1,train_set1, ARD = True)
                
                
                
        model = GPy.models.GPClassification(train_set1,Y_train,kernel_train)
                
        model.optimize(messages=False,max_f_eval = 2000)
        
        model.optimize_restarts(num_restarts = 5, verbose=False)
        
        #params[i,:] = model.param_array
        ard_coef[i,:] = kernel_train.variances.values
        #print(np.sum(params[i,:] - np.abs(params[i,:])))
        
        
                
                
        #pred_train = model.predict(train_set1)
                
        pred_test = model.predict(test_set1)
                
                
        if pred_test[0] >= 0.5:
            new_pred_test = 1
        else:
            new_pred_test = 0
        #pred_train = pred_train.flatten()
        #Y_train = Y_train.flatten()
        print("Data: {} and {}".format(new_pred_test, Y_test))
                
                
                
        
        #train_acc[i] = metrics.accuracy_score(Y_train,pred_train)
                
        
        #Y_test = Y_test.flatten()
        #new_pred_test = new_pred_test.flatten()
        Y_test = np.array(Y_test)
        new_pred_test = np.array(new_pred_test)
        Y_test = Y_test.flatten()
        new_pred_test = new_pred_test.flatten()
                
        test_acc[i] = metrics.accuracy_score(Y_test,new_pred_test)
                
                
    #train_acc_tot = np.mean(train_acc)
    #print(train_acc_tot)
    test_acc_tot = np.mean(test_acc)
    print(test_acc_tot)
    media = np.mean(ard_coef, axis=0)
    varianza = np.var(ard_coef,axis=0)
    #media_ard = np.mean(params, axis=0)
    #varianza_ard = np.var(params,axis=0)
    np.save(r'/export/usuarios01/abelenguer/Datos/media_white_noard.npy', media)
    np.save(r'/export/usuarios01/abelenguer/Datos/varianza_white_noard.npy', varianza)
    #np.save(r'/export/usuarios01/abelenguer/Datos/media_ard_white.npy', media_ard)
    #np.save(r'/export/usuarios01/abelenguer/Datos/varianza_ard_white.npy', varianza_ard)
    np.save(r'/export/usuarios01/abelenguer/Datos/data_pos1_cef.npy',data_pos1)
    np.save(r'/export/usuarios01/abelenguer/Datos/data_pos1_white.npy', data_pos2)
    np.save(r'/export/usuarios01/abelenguer/Datos/cut.npy',cut)
#        acc_train.append(train_acc_tot)
#        acc_test.append(test_acc_tot)
#        nums[i,0]=x
#        nums[i,1]=y
#        nums[i,2]=z

if __name__ == "__main__":
    main3(r'/export/usuarios01/abelenguer/Datos/RM_data/T2-MRI_white_matter-risperidona',r'/export/usuarios01/abelenguer/Datos/Mascaras_atlas/WM_mask.nii',r'/export/usuarios01/abelenguer/Datos/RM_data/T2_MRI_cerebrospinal_fluid_risperidona',r'/export/usuarios01/abelenguer/Datos/Mascaras_atlas/csf_mask.nii')