# -*- coding: utf-8 -*-
"""
@file      :  0831_thread_gen_1khz.py
@Time      :  2022/8/31 10:02
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""
# nohup python 0831_thread_gen_1khz.py  >> /mnt/sda/xbj/thread_0831_gen_data.log 2>&1 &

import datetime
import os
import threading


def execCmd(cmd):
    try:
        print("COMMAND -- %s -- BEGINS -- %s -- " % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("COMMAND -- %s -- ENDS -- %s -- " % (cmd, datetime.datetime.now()))
    except:
        print("Failed -- %s -- " % cmd)


# 如果只是路径变了的话，就改这3个地方,
# Don't forget the last '/' in those paths!!!!
# Carefully check!!!





# config_path  = [dir_str_head + "arthur-sykes-rymer-auditorium-university-york",
#            dir_str_head + "creswell-crags",
#            dir_str_head + "elveden-hall-suffolk-england",
#            dir_str_head + "gill-heads-mine",
#            dir_str_head + "hoffmann-lime-kiln-langcliffeuk",
#            dir_str_head + "innocent-railway-tunnel",
#            dir_str_head + "koli-national-park-summer",
#            dir_str_head + "koli-national-park-winter",
#            dir_str_head + "ron-cooke-hub-university-york",
#            dir_str_head + "york-guildhall-council-chamber",
#            ]





MIC_CONFIG  = [ "arthur-sykes-rymer-auditorium-university-york", "central-hall-university-york","creswell-crags", "elveden-hall-suffolk-england",
                 "gill-heads-mine", "hoffmann-lime-kiln-langcliffeuk", "innocent-railway-tunnel", "koli-national-park-summer", "koli-national-park-winter",
                "Miscellaneous", "Recreation", "ron-cooke-hub-university-york", "spring-lane-building-university-york","Underground","Underpasses","Venues",
                "york-guildhall-council-chamber",
           ]



config_path  = [ "arthur-sykes-rymer-auditorium-university-york", "central-hall-university-york","creswell-crags", "elveden-hall-suffolk-england",
                 "gill-heads-mine", "hoffmann-lime-kiln-langcliffeuk", "innocent-railway-tunnel", "koli-national-park-summer", "koli-national-park-winter",
                "Miscellaneous", "Recreation", "ron-cooke-hub-university-york", "spring-lane-building-university-york","Underground","Underpasses","Venues",
                "york-guildhall-council-chamber",
           ]





if __name__ == "__main__":
   

    commands_1 = ["python option.py --need_config " + config_path[i] + " --MIC_CONFIGs " +  MIC_CONFIG[i]  for i in range(len(config_path))]  
    command_2 = ["python main.py"]
    threads = []
    for cmd in commands_1:
        th = threading.Thread(target=execCmd, args=(cmd,))
        th.start()
        threads.append(th)
        for cmd_1 in command_2:
            th2 = threading.Thread(target=execCmd, args=(cmd_1,))
            th2.start()
            threads.append(th2)
    for th in threads:
        th.join()
            



       
        
        print("Finished!!!")
        
     
        
        
        
        
        
        
        



