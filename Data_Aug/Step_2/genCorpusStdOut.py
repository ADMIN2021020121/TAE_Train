

def genCorpusStdOut(results, params):
    #% CONFIG
    VERSION                 = 1

    # CONSTANTS
    FLOAT_FMT               = "%.10g"
    INT_FMT                 = "%d"
    STR_FMT                 = "%s"
    OUT_DEL_FMT             = ", "
    HEAD_DEL_FMT = ":"
    HEADING_FMT_list = [
                   "Test ID" ,           #HEAD_DEL_FMT...01
                   "Ver"   ,           # HEAD_DEL_FMT...02
                   "fs" ,               #HEAD_DEL_FMT...03
                   "Room" ,             #HEAD_DEL_FMT...04
                   "Room Config",       #HEAD_DEL_FMT...05
                   "Channel",           #HEAD_DEL_FMT...06
                   "Freq band" ,       # HEAD_DEL_FMT...07
                   "Centre freq",       #HEAD_DEL_FMT...08
                   "DRR"  ,            # HEAD_DEL_FMT...09
                   "DRR Mean (Ch)",     #HEAD_DEL_FMT...10
                   "T60"   ,           # HEAD_DEL_FMT...11
                   "T60 Mean (Ch)" ,    #HEAD_DEL_FMT...12
                   "FB DRR "  ,         #HEAD_DEL_FMT...13
                   "FB DRR Mean (Ch)", # HEAD_DEL_FMT...14
                   "FB T60"  ,          #HEAD_DEL_FMT...15
                   "FB T60 Mean (Ch)", # HEAD_DEL_FMT...16
                   "Filename"          #HEAD_DEL_FMT...17

                   ]
    if results["testID"] >1:
        HEADING_FMT = ""
    else:
        HEADING_FMT_list = [HEADING_FMT_list[i]+HEAD_DEL_FMT if not i==len(HEADING_FMT_list)-1 else HEADING_FMT_list[i] for i in range(len(HEADING_FMT_list)) ]
        HEADING_FMT = ", ".join(HEADING_FMT_list)
    # OUTPUT_FMT = [
    #               INT_FMT ,            #OUT_DEL_FMT...01
    #               INT_FMT ,           # OUT_DEL_FMT...02
    #               INT_FMT ,            ##OUT_DEL_FMT...03
    #               STR_FMT ,           # OUT_DEL_FMT...04
    #               STR_FMT  ,          # OUT_DEL_FMT...05
    #               INT_FMT ,            #OUT_DEL_FMT...06
    #               INT_FMT  ,           #OUT_DEL_FMT...07
    #               FLOAT_FMT ,          #OUT_DEL_FMT...08
    #               FLOAT_FMT ,         # OUT_DEL_FMT...09
    #               FLOAT_FMT ,          #OUT_DEL_FMT...10
    #               FLOAT_FMT ,          #OUT_DEL_FMT...11
    #               FLOAT_FMT ,         # OUT_DEL_FMT...12
    #               FLOAT_FMT ,          #OUT_DEL_FMT...13
    #               FLOAT_FMT ,          #OUT_DEL_FMT...14
    #               FLOAT_FMT ,         # OUT_DEL_FMT...15
    #               FLOAT_FMT ,          #OUT_DEL_FMT...16
    #               STR_FMT  ,         # OUT_DEL_FMT...17
    #               "\n"
    #               ]
    # if results["testID"] == 1:
    #     outputLine = [ s for s in HEADING_FMT][0]
    # else:
    #     outputLine = ""
    #TODO 因为不太清楚它这句输出想做什么，所以不确定改的对不对
    # res = [
    #      OUTPUT_FMT,
    #     results["testID"],                 #01
    #     VERSION,                 #02
    #     params.fs,               # 03
    #     params.roomCodeName,               # 04
    #     params.roomConfig,                 #05
    #     results["channel"],                #06
    #     results["freqBand"],                #07
    #     results["centreFreq"],                #08
    #     results["DRR"],                #09
    #     results["DRRMean"],               #10
    #     results["T60"],                #11
    #     results["T60Mean"],                 #12
    #     results["DRRFullband"],               #13
    #     results["DRRFullbandMean"],                #14
    #     results["T60Fullband"],                 #15
    #     results["T60FullbandMean"],                #16
    #     results["fileName"]               #17
    # ]

    res_list = [
        "%d" %(results["testID"]),
        "%d" %(VERSION),
        "%d" %(params.fs),
        "%s" %(params.roomCodeName),
        "%s" %(params.roomConfig),
        "%d" %(results["channel"]),
        "%d" %(results["freqBand"]),
        "%f" %(results["centreFreq"]),
        "%f" %(results["DRR"]),
        "%f" %(results["DRRMean"]),
        "%f" %(results["T60"]),
        "%f" %(results["T60Mean"]),
        "%f" %(results["DRRFullband"]),
        "%f" %(results["DRRFullbandMean"]),
        "%f" %(results["T60Fullband"]),
        "%f" %(results["T60FullbandMean"]),
        "%s" %(results["fileName"])

    ]
    # outputLine = outputLine + res
    res = ", ".join(res_list)
    if results["testID"] ==1:
        return HEADING_FMT_list,res_list#HEADING_FMT +", "+ res
    else:
        return None,res_list