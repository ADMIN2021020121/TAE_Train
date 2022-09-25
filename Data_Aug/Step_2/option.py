import argparse

parser = argparse.ArgumentParser(description='load these files')
#
parser.add_argument('--CORPUS_INPUT_FOLDER_ROOT', default='/data2/wzd/0921_STI/Rir_dataset/', type=str,help='load rir root')
parser.add_argument('--CORPUS_OUTPUT_FOLDER_ROOT', default='/data2/wzd/0921_STI/OUtput_Wav_Train/', type=str)
parser.add_argument('--T60DRRresultsFile', default='/data2/wzd/0921_STI/Rir_dataset/0921_STI.csv', type=str)
#为了加速训练，必需按照config分开才行


parser.add_argument('--need_config', default='Six_Config', type=str)

 


parser.add_argument('--MIC_CONFIGs', default="Six_Config", type=str)
#parser.add_argument('--MIC_CONFIGs', default="Nature,Miscellaneous,Recreation,Stairwells,Underground,Underpasses,Venues", type=str)
parser.add_argument('--noise_dir', default="/data2/cql/code/augu_data/get_data/15NoiseScenes_txt", type=str)
parser.add_argument('--timit_root', default='/data2/cql/code/augu_data/train_data_TIMIT', type=str)
parser.add_argument('--Speaker_txt', default='/data2/wzd/0921_STI/a.txt', type=str)
args = parser.parse_args()
