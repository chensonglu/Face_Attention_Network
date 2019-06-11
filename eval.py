import argparse

import torch
from torch import nn
from torchvision import transforms

from csv_eval import evaluate
from dataloader import WIDERDataset, AspectRatioBasedSampler, collater, Resizer, Augmenter, Normalizer, CSVDataset

from torchviz import make_dot, make_dot_from_trace
from tensorboardX import SummaryWriter
from torchsummary import summary

import torch.onnx
from torch.autograd import Variable

from model_level_attention import ResNet
from model_level_attention import resnet18, resnet34, resnet50, resnet101, resnet152


is_cuda = torch.cuda.is_available()
print('CUDA available: {}'.format(is_cuda))

ckpt = False


def dict_slice(ori_dict, start, end):
    """
    字典类切片
    :param ori_dict: 字典
    :param start: 起始
    :param end: 终点
    :return:
    """
    slice_dict = {k: ori_dict[k] for k in list(ori_dict.keys())[start:end]}
    return slice_dict


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model', help='Name of the model to load')

    parser = parser.parse_args(args)

    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                             transform=transforms.Compose([Resizer(), Normalizer()]))


    retinanet = torch.load(parser.model).cuda()
    retinanet.training = False

    # 需要确定是否是因为最后几层产生了list导致的bug，如果是只取前面若干层用来绘图
    # net_part = nn.Sequential(*list(retinanet.children())[:-9])
    # print(retinanet)

    # summary
    # summary(net_part, input_size=(3, 832, 1280))

    # dummy_input = Variable(torch.randn(1,3,832,1280)).cuda()
    # torch.onnx.export(retinanet, dummy_input, "model.onnx")

    mAP = evaluate(dataset_val, retinanet, is_cuda=is_cuda)

    print(mAP)

if __name__ == '__main__':
    main()
