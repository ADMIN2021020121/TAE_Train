import operator
import argparse
import wave
import numpy as np
import math
import random
#import splweighting
#import gtg
# from torchvision import transforms, utils
#from data_load import Rescale, RandomCrop, Normalize, ToTensor
#import mymodel as models
# import torch
import glob
import pandas as pd
import os

# def resample(waveFile):
#     dir_str = r"/data2/cql/code/cnnLstmPredictT60/声源/20220123_威力塔斯/教室1/电脑录音"
#     save_dir = "./声源/test_resample/"
#     name = file_name.split("/")[-1]
#     new_file_name = "toWav" + "_" + name.split(".")[0] + "22.wav"
#     os.system("ffmpeg -i %s -acodec pcm_s16le -ar 16000 %s" % (file_name, save_dir + new_file_name))
# sourceRIRFileName = r'E:/yousonic_code/ACE Chanllege/Lin8Ch/Office_2/1/Lin8Ch_803_1_RIR.wav'
# resample(sourceRIRFileName)


def strcmp(str1,str2):
    # i = 0
    # while i < len(str1) and i < len(str2):
    #     outcome = operator.ne(str1[i], str2[i])
    #     if outcome:
    #         print(outcome)
    #         return outcome
    #     i += 1
    # return operator.ne(len(str1), len(str2))
    if str1==str2:
        return True
    else:
        return False

def judgeList(list1,list2):

    res = []
    if len(list1) != len(list2):
        for i in range(len(list1)):
            res.append(list1[i]==list2[0])
    else:
        for i in range(len(list1)):
            res.append(list1[i]==list2[i])
    return res

def v_addnoise(Voice, noise, snr):
    Data = Voice
    data = noise
    data = np.tile(data,10)
    # if fs == Fs and len(Data)<=len(data):
    Average_Energy = np.sum(Data ** 2) / len(Data)
    average_energy = np.sum(data ** 2) / len(data)
    k = math.sqrt(Average_Energy / average_energy / 10 ** (snr * 0.1))
    print(Average_Energy, average_energy, k)
    num = random.randint(16000,len(data)-len(Data)-16000)
    Data_new = Data + data[num:len(Data)+num] * k
    return Data_new
    #     print(Vn)
    #     sf.write(Vn, Data_new, 8000)
    #     return Data, data, Data_new
    # else:
    #     print('error: fs!=Fs or len(Voice)>len(noise)')
    #     return 0, 0, 0

