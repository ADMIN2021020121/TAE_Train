from getACECorpusData import getACECorpusData
from genACECorpusDataset import genACECorpusDataset
from option  import args

class Config():
    def __init__(self,ACECorpusData):
        self.TESTNAME = 'test_gen_corpus_dataset'
        self.STARTOFFSET = 0
        self.STARTOFFSETDATE = ''

        # config
        self.NO_WRITE_MODE = 0
        self.OVERWRIE_WAV_FILES = 0
        self.READ_FROM_SERVER = 0
        self.WRITE_TO_SERVER_ANYWAY = 1

        self.DATASET = 0  # 0 for Dev,1 for Eval
        self.DATASET_TYPE = [
            ['Dev', ACECorpusData.DIST_FS, ACECorpusData.DIST_BITS_PER_SAMPLE, ACECorpusData.DEV_MIC_CONFIG_RANGE,
             ACECorpusData.ROOM_MIC_DIST_DEV_RANGE, \
             ACECorpusData.DEV_TALKER_RANGE, ACECorpusData.DEV_UTTER_RANGE, ACECorpusData.DEV_NOISES_RANGE,
             ACECorpusData.SNRs_DEV, ACECorpusData.REC_CONFIG_PREFIX_LIN8CH, ACECorpusData.SINGLE_CHANNEL_GT_DEV],
            ['Eval', ACECorpusData.DIST_FS, ACECorpusData.DIST_BITS_PER_SAMPLE, ACECorpusData.EVAL_MIC_CONFIG_RANGE,
             ACECorpusData.ROOM_MIC_DIST_EVAL_RANGE, \
             ACECorpusData.EVAL_TALKER_RANGE, ACECorpusData.EVAL_UTTER_RANGE, ACECorpusData.EVAL_NOISES_RANGE,
             ACECorpusData.SNRs_EVAL, ACECorpusData.REC_CONFIG_PREFIX_CRUCIF, ACECorpusData.SINGLE_CHANNEL_GT_EVAL]
        ]
        self.DATASET_NAME_POS = 1 - 1
        self.DATASET_FS_POS = 2 - 1
        self.DATASET_BITS_PER_SAMPLE_POS = 3-1
        self.DATASET_MIC_POS = 4 - 1
        self.DATASET_ROOMMIC_POS = 5 - 1
        self.DATASET_TALKER_POS = 6 - 1
        self.DATASET_UTTER_POS = 7 - 1
        # config['DATASET_UTTER_POS'] =
        self.DATASET_NOISE_POS = 8 - 1
        self.DATASET_SNR_POS = 9 - 1
        self.DATASET_SINGLE_MIC_GT_POS = 10 - 1
        self.DATASET_SINGLE_MIC_GT_CHAN_POS = 11 - 1

        self.params = dict()
        self.datasetName = self.DATASET_TYPE[self.DATASET][self.DATASET_NAME_POS]
        self.fs = self.DATASET_TYPE[self.DATASET][self.DATASET_FS_POS]
        self.bitsPerSample = self.DATASET_TYPE[self.DATASET][
            self.DATASET_BITS_PER_SAMPLE_POS]
        self.micConfigRange = self.DATASET_TYPE[self.DATASET][self.DATASET_MIC_POS]
        self.roomMicDistRange = self.DATASET_TYPE[self.DATASET][self.DATASET_ROOMMIC_POS]
        self.talkerRange = self.DATASET_TYPE[self.DATASET][self.DATASET_TALKER_POS]
        self.utterRange = self.DATASET_TYPE[self.DATASET][self.DATASET_UTTER_POS]
        self.noiseRange = self.DATASET_TYPE[self.DATASET][self.DATASET_NOISE_POS]
        self.snrRange = self.DATASET_TYPE[self.DATASET][self.DATASET_SNR_POS]
        self.singleMicConfigGT = self.DATASET_TYPE[self.DATASET][
        self.DATASET_SINGLE_MIC_GT_POS]
        self.singleMicConfigGTChan = self.DATASET_TYPE[self.DATASET][
        self.DATASET_SINGLE_MIC_GT_CHAN_POS]

        self.testName = self.TESTNAME
        self.startOffset = self.STARTOFFSET
        self.startOffsetDate = self.STARTOFFSETDATE
        self.readFromServer = self.READ_FROM_SERVER
        self.overwriteWavFiles = self.OVERWRIE_WAV_FILES
        self.noWriteMode = self.NO_WRITE_MODE
        ismac = 0
        #服务器数据地址
        # result 是ACE方面的数据集
        # result1 是将语音文件换成了TIMIE
        # result2 是将timit + TUT数据集来生成语音文件
        #result3 是rir+timit+TUT的结果文件
        #对于result4,因为result3生成的数据太多了，所以用result4表示最后生成的数据
        #result5 是进行多线程运算
        #result7 是用于生成部分数据
        #result8 用于生成全部的数据,也是按rir不同config进行划分,timit有526,noise15,rir 128
        # CORPUS_INPUT_FOLDER_ROOT = '/data2/queenie/IEEE2015Ace/'       #queenie debug
        # CORPUS_OUTPUT_FOLDER_ROOT =  '/data2/cql/code/IEEE2015Ace_test/Data/result2/'    #queenie debug

        # #新的rir文件
        # CORPUS_INPUT_FOLDER_ROOT = '/data2/cql/code/augu_data/EchoThiefImpulseResponseLibrary_test/'  # queenie debug
        # CORPUS_OUTPUT_FOLDER_ROOT = '/data1/cql/test_icothief/Data/result0/'  # result2是我要新生成的-2022-05-12
        # 之前那一版，没有考虑同一个说话者，所以重新生成
        # CORPUS_INPUT_FOLDER_ROOT = '/data2/cql/code/augu_data/test_icothief/split_icothief_1/'
        # CORPUS_OUTPUT_FOLDER_ROOT = '/data3/cql1/icothief/test/Data/result0/'  # result2是我要新生成的-2022-05-12
        CORPUS_INPUT_FOLDER_ROOT = args.CORPUS_INPUT_FOLDER_ROOT
        CORPUS_OUTPUT_FOLDER_ROOT = args.CORPUS_OUTPUT_FOLDER_ROOT
        # if ismac: result4是有分开生成的，result5测试,result6合并生成
        #     CORPUS_INPUT_FOLDER_ROOT = '/Volumes/ACE/Distribution/Corpus/'#% Must
        #
        #
        # else:
        #     # CORPUS_INPUT_FOLDER_ROOT = '/mnt/ACE/Distribution/Corpus/' #% Must
        #     #CORPUS_OUTPUT_FOLDER_ROOT = '/mnt/ACE/Distribution/'

        # CORPUS_INPUT_FOLDER_ROOT = 'E:/yousonic_code/ACE Chanllege/'  # % Must
        # CORPUS_OUTPUT_FOLDER_ROOT = 'E:/yousonic_code/ACE Chanllege/Data/result1/'

        self.corpusOutputFolderRoot = CORPUS_OUTPUT_FOLDER_ROOT
        self.corpusInputFolderRoot = CORPUS_INPUT_FOLDER_ROOT


if __name__=="__main__":
    #config = dict()
    ACECorpusData = getACECorpusData(0)
    config = Config(ACECorpusData)

    genACECorpusDataset(config)
    #configure the output files
    # config['TESTNAME'] = 'test_gen_corpus_dataset'
    # config['STARTOFFSET'] = 0
    # config['STARTOFFSETDATE'] = ''
    #
    # #config
    # config['NO_WRITE_MODE'] = 0
    # config['OVERWRIE_WAV_FILES'] = 0
    # config['READ_FROM_SERVER'] = 0
    # config['WRITE_TO_SERVER_ANYWAY'] = 1
    #
    # config['DATASET'] = 1    #0 for Dev,1 for Eval
    # config['DATASET_TYPE'] = [
    #     ['Dev',ACECorpusData.DIST_FS,ACECorpusData.DIST_BITS_PER_SAMPLE,ACECorpusData.DEV_MIC_CONFIG_RANGE,ACECorpusData.ROOM_MIC_DIST_DEV_RANGE,\
    #      ACECorpusData.DEV_TALKER_RANGE,ACECorpusData.DEV_UTTER_RANGE,ACECorpusData.DEV_NOICES_RANGE,ACECorpusData.SNRs_DEV,ACECorpusData.REC_CONFIG_PREFIX_LIN8CH,ACECorpusData.SINGLE_CHANNEL_GT_DEV],
    #     ['Eval',ACECorpusData.DIST_FS,ACECorpusData.DIST_BITS_PER_SAMPLE,ACECorpusData.EVAL_MIC_CONFIG_RANGE,ACECorpusData.ROOM_MIC_DIST_EVAL_RANGE,\
    #      ACECorpusData.EVAL_TALKER_RANGE,ACECorpusData.EVAL_UTTER_RANGE,ACECorpusData.EVAL_NOISES_RANGE,ACECorpusData.SNRs_EVAL,ACECorpusData.REC_CONFIG_PREFIX_CRUCIF,ACECorpusData.SINGLE_CHANNEL_GT_EVAL]
    # ]
    # config['DATASET_NAME_POS'] = 1-1
    # config['DATASET_FS_POS'] = 2-1
    # config['DATASET_BITS_PER_SAMPLE_POS'] = 3-1
    # config['DATASET_MIC_POS'] = 4-1
    # config['DATASET_ROOMMIC_POS'] = 5-1
    # config['DATASET_TALKER_POS'] = 6-1
    # config['DATASET_UTTER_POS'] = 7-1
    # #config['DATASET_UTTER_POS'] =
    # config['DATASET_NOISE_POS'] = 8-1
    # config['DATASET_SNR_POS'] = 9-1
    # config['DATASET_SINGLE_MIC_GT_POS'] = 10-1
    # config['DATASET_SINGLE_MIC_GT_CHAN_POS'] = 11-1
    #
    #
    # config['params'] = dict()
    # config['params']['datasetName'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_NAME_POS']]
    # config['params']['fs'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_FS_POS']]
    # config['params']['bitsPerSample'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_BITS_PER_SAMPLE_POS']]
    # config['params']['micConfigRange'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_MIC_POS']]
    # config['params']['roomMicDistRange'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_ROOMMIC_POS']]
    # config['params']['talkerRange'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_TALKER_POS']]
    # config['params']['utterRange'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_UTTER_POS']]
    # config['params']['noiseRange'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_NOISE_POS']]
    # config['params']['snrRange'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_SNR_POS']]
    # config['params']['singleMicConfigGT'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_SINGLE_MIC_GT_POS']]
    # config['params']['singleMicConfigGTChan'] = config['DATASET_TYPE'][config['DATASET']][config['DATASET_SINGLE_MIC_GT_POS']]
    #
    # config['params']['testName'] = config['TESTNAME']
    # config['params']['startOffset'] = config['STARTOFFSET']
    # config['params']['startOffsetDate'] = config['STARTOFFSETDATE']
    # config['params']['readFromServer'] = config['READ_FROM_SERVER']
    # config['params']['overwriteWavFiles'] = config['OVERWRIE_WAV_FILES']
    # config['params']['noWriteMode'] = config['NO_WRITE_MODE']
    #config['params']['corpusOutputFolderRoot'] = config['CORPUS_']




