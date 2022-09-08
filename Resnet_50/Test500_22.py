import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

import torchvision.transforms

# from workspace_utils import active_session
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')
import valdata_meant60
from valdata_meant60 import Val_meanT60
from torch.utils.tensorboard import SummaryWriter
# 决定使用哪块GPU进行训练
import csv
import os
import glob
from new_data_load import Dataset_dict, collate_fn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device_ids = [0, 1]
## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
# from Resnet_Model import ResNet
from Resnet_50_model import ResNet
from Bottle_neck import Bottleneck
# from mymodel import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim
import datetime
from valdata_meant60 import ValDataset, Val_meanT60
import argparse

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('--trained_epoch', type=int, default=0)

    parser.add_argument('--save_dir', type=str, default='save_model/CNN/train_acewithtimit_05262_freq_8_28')

    parser.add_argument('--model_path', type=str, default="/data2/wzd/0829_1000Hz_Dataset/TAE_500Hz_Resnet_1000/save_model/t60_predict_model_6_fullT60_rir_timit_noise.pt" )
    parser.add_argument('--epoch_for_save', type = int, default=6)
    parser.add_argument('--test_path', type=str, default="/data2/wzd/0829_1000Hz_Dataset/0830_1000_Pt/val")
    parser.add_argument('--load_pretrain', type=bool, default=True)

    parser.add_argument('--start_freq', type=int, default=7)
    parser.add_argument('--end_freq', type=int, default=22)

    parser.add_argument('--image', type=int, default=3)  # 0  1   2  3  4  5
    parser.add_argument('--gt', type=int, default=9)  # 0   3   6   9    12   15

    args = parser.parse_args()
    return args
## The code is used to predict T60 among 125  250 500 1000 2000 4000
## You only modify the image and gt of the parse_args function

def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend(fontsize=20)


def get_boxplot(test_res):
    room_dict = {}

    for key, value in test_res.items():
        if '.pt' not in key:
            continue
        room_config, room_name = key.split('/')[-2:]
        temp_room = room_name.split(room_config)[1].strip('_')  # 'mine_site1_1way_bformat_2'

        if temp_room not in room_dict.keys():
            # then create a new room_dict
            # room_dict[temp_room] = {'gt': value['gt'].tolist(), 'mean_output': value['mean_output'].tolist(),
            #                         'mean_bias': value['mean_bias'].tolist()}
            room_dict[temp_room] = {'gt': [value['gt'].tolist()], 'mean_output': [value['mean_output'].tolist()],
                                    'mean_bias': value['mean_bias'].tolist()}

        else:
            # room_dict[temp_room]['gt'].extend(value['gt'].tolist())
            room_dict[temp_room]['gt'].extend([value['gt'].tolist()])
            # room_dict[temp_room]['mean_output'].extend(value['mean_output'].tolist())
            room_dict[temp_room]['mean_output'].extend([value['mean_output'].tolist()])
            room_dict[temp_room]['mean_bias'].extend(value['mean_bias'].tolist())

    """"aaa"""
    fig = plt.figure(figsize=(40, 20))
    ticks = list(room_dict.keys())
    for i in range(len(ticks)):
        if i // 2 == 0:
            ticks[i] = '\n' + ticks[i]
    gt_plot = plt.boxplot([v['gt'] for _, v in room_dict.items()],
                          positions=np.array(np.arange(len(room_dict.values()))) * 2.0-0.3, widths=0.3)
    mean_output_plot = plt.boxplot([v['mean_output'] for _, v in room_dict.items()],
                                   positions=np.array(np.arange(len(room_dict.values()))) * 2.0 + 0.3, widths=0.3)
    #mean_bias_plot = plt.boxplot([v['mean_bias'] for _, v in room_dict.items()],
    #                             positions=np.array(np.arange(len(room_dict.values()))) * 2.0 + 0.6, widths=0.3)

    # setting colors for each groups
    define_box_properties(gt_plot, 'black', 'gt')
    define_box_properties(mean_output_plot, '#D7191C', 'mean_output')
    # define_box_properties(mean_bias_plot, '#2C7BB6', 'mean_bias')

    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, fontsize=12)
    plt.yticks(fontsize=20)
    # plt.tight_layout()
    # set the title
    # plt.xticks(rotation=-45)
    freq_name = '500Hz'
    fig_name = 'Test_' + freq_name + '_epoch' + str(args.epoch_for_save)
    plt.title(fig_name, fontsize=30)  # 标题，并设定字号大小
    plt.savefig(outputresult_dir + '/' + fig_name + '.png', dpi=fig.dpi, pad_inches=4)
    plt.show()





def load_checkpoint(checkpoint_path=None, trained_epoch=None, model=None, device=None):
    save_model = torch.load(checkpoint_path, map_location=device)
    # model_dict = model.state_dict()
    # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

    # model_dict.update(state_dict)

    # model.load_state_dict(model_dict)
    model.load_state_dict(save_model['model'])
    trained_epoch_load = save_model['epoch']
    # trained_epoch = state['epoch']
    print('model loaded from %s' % checkpoint_path)
    return_epoch = 0
    if not trained_epoch is None:
        return_epoch = trained_epoch
    else:
        return_epoch = trained_epoch_load

    return model, return_epoch


def output_result_analysis(result_dict, output_dir, test_path):
    test_dataset = test_path.split("/")[-1]
    csv_file = os.path.join(output_dir, test_dataset + ".csv")
    margin = 0.2
    larget_than_margin_csv = csv_file.split(".")[0] + "larger_than_%f" % (margin) + ".csv"

    f = open(csv_file, "w", newline='')
    csv_writer_normal = csv.writer(f)

    f_larger = open(larget_than_margin_csv, "w")
    larger_csv_writer = csv.writer(f_larger)
    for key, value in result_dict.items():
        # print('key', key)

        csv_writer = csv_writer_normal


        # 改
        csv_writer.writerow([0, 500])
        csv_writer.writerow([str(key)])

        for k in range(len(result_dict[key]["output_list"])):
            csv_writer.writerow(["output%d" % (k)] + result_dict[key]["output_list"][k].cpu().numpy().tolist())


        csv_writer.writerow(["mean_output"] + [float(result_dict[key]["mean_output"])])
        csv_writer.writerow(["gt"] + [float(result_dict[key]["gt"])])
        csv_writer.writerow(["mean_bias"] + [float(result_dict[key]["mean_bias"])])
        csv_writer.writerow(["mean_mse"] + [float(result_dict[key]["mean_bias"] ** 2)])

        csv_writer.writerow([])


def test_net(net, epoch, val_loader, writer):
    result_dict = dict()
    with torch.no_grad():
        net.eval()
        total_mean_loss = torch.zeros((1, 1))
        total_mean_bias = torch.zeros((1, 1))
        progress_bar = tqdm(val_loader)
        useless_count = 0
        for j, datas in enumerate(progress_bar):

            meanT60 = datas['MeanT60'].to(torch.float32).to(device)
            images = datas['image'].to(torch.float32).to(device)
            gt_t60 = datas['t60'].to(torch.float32).to(device)
            valid_len = datas['validlen']
            names = datas['name']




            images = images.squeeze(1)
            feature = images.transpose(0, 1)[args.image].unsqueeze(1)




            output_pts = net(feature.to(device))


            gt_t60_reshape = datas['t60'].to(torch.float32).to(device)





            gt = gt_t60_reshape.transpose(0, 1)[args.gt].unsqueeze(1)


            bias = gt  - output_pts

            rsquare_error = torch.sqrt(bias ** 2)
            abs_bias = torch.abs(gt - output_pts)
            mean_output_list = []

            if not torch.isnan(rsquare_error).all():
                total_mean_loss += torch.mean(rsquare_error, dim=0).cpu().detach()
                total_mean_bias += torch.mean(bias, dim=0).cpu().detach()
            else:
                useless_count += 1

            for i in range(len(valid_len)):
                start_num = 0
                if i > 0:
                    start_num = sum(valid_len[0:i])

                output_list = [output_pts[k] for k in range(start_num, start_num + valid_len[i])]
                # mean_output = torch.mean(output_pts[start_num:start_num + valid_len[i]], dim=0)
                mean_abs_bias = torch.mean(abs_bias[start_num:start_num + valid_len[i]], dim=0)
                mean_output = torch.mean(output_pts[start_num:start_num + valid_len[i]])
                mean_bias = torch.mean(bias[start_num:start_num + valid_len[i]], dim=0)
                mean_rsquare_error = torch.mean(rsquare_error[start_num:start_num + valid_len[i]], dim=0)
                # mean_gt = torch.mean(gt_t60_reshape[start_num:start_num + valid_len[i]], dim=0)[1]
                mean_gt = torch.mean(gt[start_num:start_num + valid_len[i]])
                result_dict[names[i][0]] = {"mean_output": mean_output, "mean_bias": mean_bias, "gt": mean_gt,
                                            "square_error": mean_rsquare_error, "output_list": output_list}

        mean_loss = total_mean_loss / (len(val_loader) - batch_size * useless_count)
        mean_bias = total_mean_bias / (len(val_loader) - batch_size * useless_count)
        if not writer is None:
            writer.add_scalar('val/mean_loss', mean_loss, epoch)
            writer.add_scalar('val/mean_bias', mean_bias, epoch)

        print("Mean loss:", mean_loss)
        print("Mean bias:", mean_bias)
        # print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" %(epoch,mean_loss,mean_bias))

        return result_dict



class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.warmup = False
        self.warmup_epochs = 10
        self.min_lr = min(lr)
        self.max_lr = max(lr)
        # self.lr = lr
        # self.lr_epochs = lr_epochs

        # self.lr_warmup = lambda epoch_in : min(self.lr)+0.5*(max(self.lr) - min(self.lr))*(1+np.cos((epoch_in-self.warmup_epochs)*PI/(2*self.warmup_epochs)))
        self.lr_warmup = lambda epoch_in: self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                    1 + np.cos((epoch_in - self.warmup_epochs) * PI / (2 * self.warmup_epochs)))
        if self.warmup == True:
            self.lr_epochs = [self.warmup_epochs] + [i + self.warmup_epochs for i in lr_epochs]
            self.lr = [self.lr_warmup] + lr
        else:
            self.lr_epochs = lr_epochs
            self.lr = lr

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        if self.warmup == True and idx == 0:
            return self.lr[idx](epoch)
        else:
            return self.lr[idx]


if __name__ == "__main__":

    args = parse_args()

    DEBUG = 0
    LOAD_PRETRAIN = args.load_pretrain  # False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path

    net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1, include_top=True)

    if LOAD_PRETRAIN == True:
        start_time = time.time()
        net, trained_epoch = load_checkpoint(model_path, 99, net, device)
        print('Successfully Loaded model: {}'.format(model_path))
        print('Finished Initialization in {:.3f}s!!!'.format(
            time.time() - start_time))
    else:
        trained_epoch = 0
    net.to(device)

    # net.to(device)
    print(net)



    val_batch_size = 100
    # val_batch_size = 1
    criterion = torch.nn.MSELoss()
    # optimizer = optim.Adam([{'params':net.parameters(),'initial_lr':0.0001}], lr=0.0001,weight_decay=0.0001)
    n_epochs = 300
    data_transform = transforms.Compose([ToTensor()])


    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}


    print(get_parameter_number(net))

    if DEBUG == 1:
        val_dict_root = "/Users/queenie/Documents/ace_dict_data/ACE"
        batch_size = 2
    else:

        val_dict_root = args.test_path
        batch_size = 50

    val_transformed_dataset = Dataset_dict(root_dir=val_dict_root, transform=data_transform, start_freq=args.start_freq,
                                           end_freq=args.end_freq)
    print("len of val dataset:", len(val_transformed_dataset))
    # print('Number of images: ', len(transformed_dataset))
    if DEBUG == 1:

        val_loader = torch.utils.data.DataLoader(val_transformed_dataset, shuffle=False, num_workers=0,
                                                 batch_size=val_batch_size, drop_last=True,
                                                 collate_fn=collate_fn)
    else:

        val_loader = torch.utils.data.DataLoader(val_transformed_dataset,
                                                 shuffle=False, num_workers=1,
                                                 batch_size=val_batch_size, drop_last=True, prefetch_factor=100,
                                                 collate_fn=collate_fn)

    print("after train loader init")
    trained_epoch = args.trained_epoch

    test_output_result = test_net(net, trained_epoch, val_loader, None)
    torch.save(test_output_result, './6_epoch_Resnet_50.pt')

    outputresult_dir = "/data2/wzd/0829_1000Hz_Dataset/TAE_500Hz_Resnet_1000/Csv/Epoch_6"

    if not os.path.exists(outputresult_dir):
        os.makedirs(outputresult_dir)

    output_result_analysis(test_output_result, outputresult_dir, val_dict_root)
    get_boxplot(test_res=test_output_result)






    # TODO 训练时要改dict，权重名字，验证时保存的文件名和路径
