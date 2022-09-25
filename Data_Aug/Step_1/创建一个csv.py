import csv
# with open('innovators.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["SN", "Name1", "Contribution"])
    # writer.writerow([1, "Linus Torvalds", "Linux Kernel"])
    # writer.writerow([2, "Tim Berners-Lee", "World Wide Web"])
    # writer.writerow([3, "Guido van Rossum", "Python Programming"])

resultsFileName = "test_icothief.csv"
resultsHandle = open(resultsFileName, "a",newline="")
csv_writer = csv.writer(resultsHandle)
headLine = ["Test ID:","Ver:","fs:", "Room:","Room Config:","Session ID:","Mic Pos:",
            "Source Pos:", "Config:","Rec Type:","RIR:","Freq band:","Centre freq:",
            "Channel:","DRR:","DRR Mean (Ch):","T60 AHM:","T30 ISO:","T20 ISO:",
            "T60 AHM Mean (Ch):","T30 ISO Mean (Ch):","T20 ISO Mean (Ch):","ISO AHM Ints:","FB DRR :","FB DRR Mean (Ch):",
            "FB T60 AHM:","FB T30 ISO:","FB T20 ISO:","FB T60 AHM Mean (Ch):","FB T30 ISO Mean (Ch):","FB T20 ISO Mean (Ch):",
            "DRR direct +:","DRR direct -:"]


csv_writer.writerow(headLine)
# resultsHandle.close()

import xlrd
import os

import shutil
read_path = "E:/yousonic_code/声源/EchoThiefImpulseResponseLibrary"
save_path = "./openair_rir_1000Hz/"
# book = xlrd.open_workbook(r"C:\Users\17579\Desktop\Rir_data\Code\Self_detection.xls")
#C:\Users\17579\Desktop\job_beauty
book = xlrd.open_workbook(r"C:\Users\17579\Desktop\Rir_data\Code\STI.xls")


sheet1 = book.sheets()[0]
# 数据总行数
nrows = sheet1.nrows
# 数据总列数
ncols = sheet1.ncols

# 获取表中第三行的数据
x = sheet1.row_values(2)
# 获取表中第二列的数据
y = sheet1.col_values(1)
#获取第五列中的第二个数据
z = sheet1.col_values(4)[1]
# t60_row = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,]
copy_info = []
test_id = 0
for i in range(1, nrows):
    values = sheet1.row_values(i)
    room = values[1]
    config = values[0]
    channel = values[4]
    fre_band = 0

    for j in range(len(values)):

        if j<10:
            continue
        else:
            if j%5==0:
                info = []
                # resultsHandle = open(resultsFileName, "a", newline="")
                # csv_writer = csv.writer(resultsHandle)
                fre_band += 1
                test_id += 1
                t60 = values[j]
                info.append(test_id) #序号
                info.append(1)
                info.append(48000)
                info.append(room)
                info.append("NAN")
                info.append(1)
                info.append(2)
                info.append(config)
                info.append("IR")
                info.append(room)
                info.append(fre_band) #表示第几个频段，它没有30个频段，所有应该设置为0
                info.append(0) #中心频段设置为0
                info.append(channel)
                info.append(0)
                info.append(0)
                info.append(t60)
                #下面内容都为0
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                info.append(0)
                copy_info = info
                if fre_band == 30:
                    info[15] = 0
                csv_writer.writerow(info)
        # if j==len(values)-1:
        #      copy_info[10] = copy_info[10] + 1 #中心频率设置为30
        #      copy_info[15] = 0
        #      csv_writer.writerow(copy_info)

resultsHandle.close()