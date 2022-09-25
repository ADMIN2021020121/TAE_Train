"""
@file      :  splice_pt.py
@Time      :  2022/9/19 09:09
@Software  :  Jupyter
@summary   :  TAE_No_Freq
@Author    :  Zedong Wu
"""
import matplotlib.pyplot as plt

import gtg
import splweighting
import wave
import glob
import os



#TAE
import torch
# load_filename
import torchaudio
import numpy as np
# split_bands
import acoustics
# take_each_bank
from acoustics.signal import bandpass
# check_band_type
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
# Hilbert
from scipy import fftpack
# butter_lowpass_filter
from scipy.signal import butter,filtfilt
# downsample
from scipy import signal





import pandas as pd
import argparse
##configuration area
chunk_length = 4
chunk_overlap = 0.5
#TODO 需要改变csv,numpydir/csv_save_path这些路径


# elveden-hall-suffolk-england


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--csv_file', type=str, default="/data2/wzd/STI/0923_sti_cat_voice/0923_STI.csv")
parser.add_argument('--dir_str', type=str, default="/data1/xbj/0923_STI/Dev/Speech/Six_Config/")
parser.add_argument('--save_dir', type=str, default="/data2/wzd/STI/Pt_dataset/train/Six_Config")


args = parser.parse_args()
save_dir = args.save_dir
if not os.path.exists(args.save_dir):
    os.makedirs(save_dir)
save_dir = args.save_dir
dir_str = args.dir_str
csv_file = args.csv_file

##functions
def SPLCal(x):
    Leng = len(x)
    pa = np.sqrt(np.sum(np.power(x, 2))/Leng)
    p0 = 2e-5
    spl = 20 * np.log10(pa / p0)
    return spl

print(dir_str)
##main loop, process eahc file in dir
# g = wave.open("clean_speech_example","rb")




class Totensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, ddr, t60, meanT60 = sample['image'], sample['ddr'], sample['t60'], sample['MeanT60']

        # image, ddr, t60 = sample['image'], sample['ddr'], sample['t60']
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).squeeze(0)

        ddr = ddr.astype(float)
        t60 = t60.astype(float)
        meanT60 = meanT60.astype(float)

        # image = image.transpose((2, 0, 1))
        return {'image': image,
                'ddr': torch.from_numpy(ddr),
                't60': torch.from_numpy(t60),
                "MeanT60": torch.from_numpy(meanT60)
               }
        #






csv_data = pd.read_csv(csv_file)



def TAE_Feature(raw_signal): 
    
    list = []
   
    raw = raw_signal.T

    Number_of_sample = len(raw_signal[0])
    print("Number_of_sample ={}".format(Number_of_sample))

    image = raw.T[0]
    TAE_First = fftpack.hilbert(image)
    TAE_First_formula = np.sqrt(image**2 + TAE_First**2)

    # Low_pass

    fs = 16000
    nyq = 0.5 * fs
    cutoff = 20
    order = 6 
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, TAE_First_formula)


    downsample_index = int(Number_of_sample/400)

    # Downsample
    downsample = signal.resample(y, downsample_index)

    list.append(downsample)

    return list



for file_name in glob.glob(dir_str+r"/*.wav"):

    waveform, t = torchaudio.load(file_name, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
    f = wave.open(file_name, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]  # 声道数, 量化位数（byte单位）, 采样频率,采样点数
    print(file_name)
    print(nchannels, sampwidth, framerate, nframes/framerate)
   
    str_data = f.readframes(nframes)  # 读取音频，字符串格式
    f.close()


    _ = np.frombuffer(str_data, dtype=np.int16)
    _.shape = -1, nchannels

    audio_time = nframes/framerate
    chan_num = 0
    count = 0
    new_file_name = (file_name.split("\\")[-1]).split(".")[0]
    new_file_name = new_file_name.split("/")[-1]
    ## process each channel of audio
    for audio_samples_np in waveform.numpy():
        #whole_audio_SPL = SPLCal(audio_samples_np)

        available_part_num = (audio_time-chunk_overlap)//(chunk_length - chunk_overlap)   #4*x - (x-1)*0.5 <= audio_time    x为available_part_num

        if available_part_num ==1:
            cut_parameters = [chunk_length]
        else:
            cut_parameters = np.arange(chunk_length, (chunk_length - chunk_overlap)*available_part_num+chunk_overlap, chunk_length)  # np.arange()函数第一个参数为起点，第二个参数为终点，第三个参数为步长（10秒）
            #  输入的为18，默认起点值为4，步长为4  [4  8  12  16]
        start_time = int(0)  # 开始时间设为0
        count = 0
        #开始存储pt文件
        dict = {}
        save_data = []
        for t in cut_parameters:
            stop_time = int(t)  # pydub以毫秒为单位工作
            start = int(start_time*framerate)
            end = int((start_time+chunk_length)*framerate)
            audio_chunk = audio_samples_np[start:end]  # 音频切割按开始时间到结束时间切割

           




            print(len(audio_chunk))
            print(audio_chunk)
            audio_chunk_new = [audio_chunk]
            audio_chunk_new =np.array(audio_chunk_new)



            ## TAE
            new_list = TAE_Feature(audio_chunk_new)

     
            #下面是应该存储的相应数据，语谱图，ddr,T60,meanT60
            chan = chan_num +  1
            config = new_file_name.split("_")[0] #+"_" + new_file_name.split("_")[1]
            if config == "dirac":
                config = new_file_name.split("_")[0]  # +"_" + new_file_name.split("_")[1]
                room = new_file_name.split(config)[1][1:-1]
            else:
                config = new_file_name.split("_")[0]  + "_"  +  new_file_name.split("_")[1]
                # room = new_file_name.split(config)[1][1:-1]
                # room = new_file_name.split("_")[2] +"_"+new_file_name.split("_")[3]
                room = new_file_name.split(config)[1].strip('_')



            print(new_file_name)

            a = (csv_data['Room:'] == room).values
            print(csv_data['Room:'])
            b = (csv_data['Room Config:'] == config).values
            c = (csv_data['Channel:'] == chan).values
            # abc = a & b & c
            abc = a & b
            # data = csv_data[a & b & c]
            # data = csv_data[a & b]
            data = csv_data[a]

            T60_data = data.loc[:, ['T60:']]

            #   FB_T60_data = (data.loc[:, ['FB T60:']]).iloc[0, 0]

            FB_T60_data = data.loc[:, ['FB T60:']]


            #  FB_T60_M_data = (data.loc[:, ['FB T60 Mean (Ch):']]).iloc[0, 0]
            FB_T60_M_data = data.loc[:, ['FB T60 Mean (Ch):']]



            DDR_each_band = np.array([0 for i in range(30)])
            T60_each_band = (T60_data.values).reshape(-1)
            MeanT60_each_band = np.array([FB_T60_data,FB_T60_M_data])

         
            
            # image = chunk_result
            image = new_list
            sample = {'image': image, 'ddr': DDR_each_band, 't60': T60_each_band, "MeanT60": MeanT60_each_band}
            # sample = {'image': image, 'ddr': DDR_each_band, 't60': T60_each_band}
            transform = Totensor()
            sample = transform(sample)

            save_data.append(sample)



            start_time = start_time + chunk_length - chunk_overlap  # 开始时间变为结束时间前1s---------也就是叠加上一段音频末尾的4s
        if len(save_data)!=0:

            pt_file_name = os.path.join(save_dir, new_file_name + '-' + str(chan_num) + '.pt')
            dict[new_file_name + '-' + str(chan_num)] = save_data
            torch.save(dict, pt_file_name)
        chan_num = chan_num + 1
    print('----------------finish----------------')
