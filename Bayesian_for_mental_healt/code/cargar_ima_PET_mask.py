# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:54:50 2021

@author: Albert
"""

import os
import numpy as np
#from nibabel.testing import data_path
import nibabel as nib
from sklearn.preprocessing import StandardScaler

#Labels: 0 for healthy and 1 for squizo

def cargar_ima_PET_mask(data_path, mask_path):

    dir_list = os.listdir(data_path)
    
    data_total = []
    
    labels = []
    
    
        
    mask_ni = nib.load(mask_path)
    mask = np.array(mask_ni.dataobj)
    
    for j in range(len(dir_list)):
    
        file_name = os.path.join(data_path, dir_list[j])
        
        name = os.path.basename(file_name)
        index = name.find('Poly')
        if index == -1:
            labels.append(0)
        else:
            labels.append(1)
        
        img = nib.load(file_name)
        data = np.array(img.dataobj)
        
        
        
        
        
        
        
        data_pos = np.where(mask == 1)
        tam = data_pos[0].shape[0]
        data_vector = np.zeros((1,tam))
        
        
        
        for i in range(tam):
            data_vector[0,i]=data[data_pos[0][i],data_pos[1][i],data_pos[2][i]]
            
#        scaler = StandardScaler()
#        data_vector = scaler.fit_transform(data_vector)
#        data_vector = scaler.transform(data_vector)
        data_total.append(data_vector)
    
    data_full = np.empty((len(data_total),tam))
    
    for i in range(len(data_total)):
        fila = data_total[i]
        data_full[i,:]= fila
    
    scaler = StandardScaler()
    data_full = scaler.fit_transform(data_full)
    data_full = scaler.transform(data_full)   
        
    
        
    
    return data_full, data_pos, labels


if __name__ == "__main__":
    cargar_ima_PET_mask(data_path = r'C:\Users\Albert\Documents\TFM\RM_data\T2-MRI_white_matter-risperidona', mask_path = r'C:\Users\Albert\Documents\TFM\Mascaras atlas\WM_mask.nii')