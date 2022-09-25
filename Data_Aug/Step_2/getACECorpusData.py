from option import args
class getACECorpusData():
    def __init__(self,margin):


        self.ACE_TITLE = '(c) Acoustic Characterization of Environments (ACE) Corpus, 2015'
        self.ACE_AUTHOR = 'J. Eaton, N. D. Gaubitch, A. H. Moore, P. A. Naylor, Imperial College London'
        self.ACE_PAPER = [
                                'J. Eaton, N. D. Gaubitch, A. H. Moore, P. A. Naylor, "The ACE Challenge - corpus description and performance evaluation" '
                                'in Proc. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA, Oct. 2015'
    
                                ]
        if margin == 0:
            server = 0
    
        #Top level folder. Folders for each data group are subfolders of this
        #% folder
        #self是否要用字典来表示
        self.ACE_DATA_PATH = 'Data/'
        self.ACE_CORPUS_PATH = 'Corpus/'
        self.ACE_DATA_PATH_INT = 'Internal/'
        self.ACE_DATA_PATH_EXT = 'Distribution/'
        self.ACE_SUBMISSIONS_FOLDER = 'Submissions/'
        self.ACE_ANALYSIS_FOLDER = 'Analysis/'
        self.ACE_SUBMISS_DECODE = 'Decode/'
        self.ACE_DATA_EXT_EVAL = 'Eval/'
        self.ACE_DATA_EXT_DEV = 'Dev/'
        self.ACE_DATA_EXT_SPEECH = 'Speech/'
        self.ACE_EXT_DOCS = 'Docs/'
        self.SW_FOLDER = 'Software/'
        self.ANECHOIC_FOLDER = 'Anechoic/'
        #Sample rate
        self.FS = 48000
        self.REC_TYPE_AMBIENT = "Ambient"
        self.REC_TYPE_FAN = 'Fan'
        self.REC_TYPE_BABBLE = 'Babble'
        #Identifiers for each row
        self.REC_CONFIG_ID_POS = 1
        self.REC_CONFIG_CHANNELS_POS = 2
        self.REC_CONFIG_PREFIX_POS = 3
        self.REC_CONFIG_TYPE_POS = 4
        self.REC_CONFIG_AUTO_IR_POS = 5
    
        self.REC_CONFIG_PREFIX_SINGLE = 'Single'
        self.REC_CONFIG_PREFIX_CRUCIF = 'Crucif'
        self.REC_CONFIG_PREFIX_MOBILE = 'Mobile'
        self.REC_CONFIG_PREFIX_LIN8CH = 'Lin8Ch'
        self.REC_CONFIG_PREFIX_CB = 'Chromebook'
        self.REC_CONFIG_PREFIX_EM32 = 'EM32'
    
        #Note that these IDs are the ones used for processing the recordings, but they are
        #% different to dataset IDs used for the challenge. See
        #% audioData.DATASET_SINGLE for example
        self.REC_CONFIG_PREFIX_CRUCIF_ID = 1
        self.REC_CONFIG_PREFIX_MOBILE_ID = 2
        self.REC_CONFIG_PREFIX_LIN8CH_ID = 3
        self.REC_CONFIG_PREFIX_CB_ID = 4
        self.REC_CONFIG_PREFIX_EM32_ID = 5
    
        self.CHANS_CRUCIF = 5
        self.CHANS_MOBILE = 3
        self.CHANS_LIN8CH = 8
        self.CHANS_EM32 = 32
        self.CHANS_CB = 2
        self.CHANS_MAX = max(
            [self.CHANS_CRUCIF,self.CHANS_MOBILE,self.CHANS_LIN8CH,self.CHANS_EM32,self.CHANS_CB])
        #Description of how each channel of the data is to be written out to audio files
        #% E.g. take channels 1:5 from the FF800 recordings and write out to the Crucif... file as .wav audio
        #% last column is whether the IR is
        # ACE CHALLENGE SPECIFIC DATA TO PACKAGING
        self.DIST_FS = 16000 #Distribution sample rate
        self.DIST_BITS_PER_SAMPLE = 16 #Distribution bits per sample
    
        #ANECHOIC SPEECH
        self.TALKER_NAME_POS = 1
        self.TALKER_CODENAME_POS = 2
    
        self.TALKERs_CORPUS = [
            ['Ben','M1'],
            ['Jeroen','M2'],
            ['Jorn' ,'M3'],
            ['Nick','M4'],
            ['Tom','M5'],
            ['Bonny','F1'],
            ['Cynthia','F2'],
            ['Hayley','F3'],
            ['Martha','F4'],
            ['Noeska','F5'],
            ['Bob','M6'],
            ['Phil','M7'],
            ['Al','M8'],
            ['Josh','M9']
        ]
        self.DEV_TALKER_RANGE = range(11,15)  #11:14
        self.EVAL_TALKER_RANGE = range(1,11)  #1:10
    
        self.UTTER_TYPE_COLOUR = 'Colour'
        self.UTTER_TYPE_PLACE = 'Place'
        self.UTTER_TYPE_LIVE = 'Live'
        self.UTTER_TYPE_WORK = 'Work'
        self.UTTER_TYPE_COUNT = 'Count'
    
        self.UTTER_TYPE_NAME_POS = 1
        self.UTTER_TYPE_CODENAME_POS = 2
    
        self.UTTER_TYPEs_CORPUS = [
            ['s1',self.UTTER_TYPE_COLOUR],
            ['s2',self.UTTER_TYPE_PLACE],
            ['s3',self.UTTER_TYPE_LIVE],
            ['s4',self.UTTER_TYPE_WORK],
            ['s5',self.UTTER_TYPE_COUNT]
        ]
        self.DEV_UTTER_RANGE = range(3,5) #3:4
        self.EVAL_UTTER_RANGE = range(1,6)  #1:5
    
        #Noise types used
        self.NOISEs = [
            self.REC_TYPE_AMBIENT, #Ambient
            self.REC_TYPE_FAN, #Fan
            self.REC_TYPE_BABBLE #Babble
        ]
        self.DEV_NOISES_RANGE = range(1-1,4-1)
        self.EVAL_NOISES_RANGE = range(1-1,4-1)
    
        #SNRs applied to the different parts of the challenge
        #Since these form part of the filename, decimals are not handled with the
        #% current code.  Stick to integers
        self.SNRs_DEV = [0,10,20]
        self.SNRs_EVAL = [-1,12,18]
        #Data used to describe the test conditions in the ground truth file
        #% supplied to participants
        self.SESSION_POS = 0
        self.ROOM_NAME_POS = 1
        self.ROOM_CODENAME_POS = 2
        self.ROOM_CONFIG_POS = 3
        self.ROOM_LIN8CH_CHANNELS_POS = 5
    
        self.ROOM_MIC_DISTs_CORPUS = [
            [41,'502','Office_1','1',8],
            [47,'502','Office_1','2',8],
            [115,'EE_lobby','Building_Lobby','1',8],
            [125,'EE_lobby','Building_Lobby','2',8],
            [9,'803','Office_2','1',8],
            [17,'803','Office_2','2',8],
            [25,'611','Meeting_Room_2','1',4],
            [33,'611','Meeting_Room_2','2',8],
            [62,'503','Meeting_Room_1','1',8],
            [67,'503','Meeting_Room_1','2',8],
            [79,'403a','Lecture_Room_2','1',8],
            [84, '403a','Lecture_Room_2','2',8],
            [97,'508','Lecture_Room_1','1',8],
            [101,'508','Lecture_Room_1','2',8]]
        self.ROOM_MIC_DIST_DEV_RANGE = range(0,4)
        self.ROOM_MIC_DIST_EVAL_RANGE = range(4,14)
        #The files containing the ground truth T60 and DRR measurements for each
        #% of the datasets
        self.CORPUS_T60_DRR_MEASUREMENTS = '20150225T195903_test_t60_DRR_measurement_results'
        #This is used in the decode algorithm only for final decyphering in the
        #% EVAL results.  502 and EE_Lobby included for completeness, although they
        #% never actually get used
        self.ROOMS = [
        ['803','Office 2'],
        ['503','Meeting Room 1'],
        ['611','Meeting Room 2'],
        ['508','Lecture Room 1'],
        ['403a','Lecture Room 2'],
        ['502','Office 1'],
        ['EE_Lobby','Building Lobby']
        ]
        #Description of the actual datasets used in the challenge
        self.DATASET_SINGLE = 1
        self.DATASET_CRUCIF = 2
        self.DATASET_MOBILE = 3
        self.DATASET_LIN8CH = 4
        self.DATASET_SPHERE = 5
        self.DATASET_LAPTOP = 6
        self.REC_CONFIG_SINGLE = 'Single'
        ###

        self.MIC_CONFIGs = [
        self.REC_CONFIG_SINGLE,
        self.REC_CONFIG_PREFIX_CRUCIF,
        self.REC_CONFIG_PREFIX_MOBILE,
        self.REC_CONFIG_PREFIX_LIN8CH,
        self.REC_CONFIG_PREFIX_EM32,
        self.REC_CONFIG_PREFIX_CB
        ]
        # self.MIC_CONFIGs = ["Miscellaneous",
        #                     "Nature",
        #                     "Recreation",
        #                     "Stairwells",
        #                     "Underground",
        #                     "Underpasses",
        #                     "Venues"]


        MIC_CONFIGs = args.MIC_CONFIGs.split(",")
        self.MIC_CONFIGs = MIC_CONFIGs
    
        self.nMIC_CONFIGs = len(self.MIC_CONFIGs)
    
        self.SINGLE_CONFIG_GT_DEV = self.REC_CONFIG_PREFIX_LIN8CH
        self.SINGLE_CONFIG_ID_GT_DEV = self.REC_CONFIG_PREFIX_LIN8CH_ID
        self.SINGLE_CHANNEL_GT_DEV = 1
        self.SINGLE_CONFIG_GT_EVAL = self.REC_CONFIG_PREFIX_CRUCIF
        self.SINGLE_CONFIG_ID_GT_EVAL = self.REC_CONFIG_PREFIX_CRUCIF_ID
        self.SINGLE_CHANNEL_GT_EVAL = 1
    
        self.DEV_MIC_CONFIG_RANGE = len(self.MIC_CONFIGs)
        self.EVAL_MIC_CONFIG_RANGE = len(self.MIC_CONFIGs)
    
        #Algorithm descriptions.  Used by partipants.  Joint estimation has to be
        #% handled manually
        self.ALG_TYPE_T60_FULL = 1
        self.ALG_TYPE_T60_SUB = 2
        self.ALG_TYPE_T60_FULL_SUB = 3
        self.ALG_TYPE_DRR_FULL = 4
        self.ALG_TYPE_DRR_SUB = 5
        self.ALG_TYPE_DRR_FULL_SUB = 6
        self.ALG_TYPE_JOINT = 7
        self.ALG_DECODE = [
        '$T_{60}$ fullband',
        '$T_{60}$ subbands',
        '$T_{60}$ full and subbands',
        'DRR fullband',
        'DRR subbands',
        'DRR full and subbands',
        'Joint estimation'
        ]
    
        #新增对TIMIT数据的路径
        self.TIMIT_TRAIN_PATH = "./train_data_TIMIT"
        self.TIMIT_TRAIN_TXT = args.Speaker_txt

        self.TIMIT_TEST_PATH = "./test_data_TIMIT"
        self.TIMIT_TESTTXT = "./test.txt"
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
