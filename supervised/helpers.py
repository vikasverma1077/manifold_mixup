'''
Created on 16 Nov 2017

'''
from time import gmtime, strftime
import torch
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt


def copy_script_to_folder(caller_path, folder):
    script_filename = caller_path.split('/')[-1]
    script_relative_path = os.path.join(folder, script_filename)
    # Copying script
    shutil.copy(caller_path, script_relative_path)

def cyclic_lr(initial_lr,step,total_steps,num_cycles):
    factor=np.ceil(float(total_steps)/num_cycles)
    theta=np.pi*np.mod(step-1,factor)/factor
    return (initial_lr/2)*(np.cos(theta)+1)

if __name__ == '__main__':
    lr_list=[]
    for i in xrange(1000):
        lr=cyclic_lr(0.1,i+1,1100,3)
        lr_list.append(lr)
    plt.plot(np.asarray(lr_list))
    plt.show()
        