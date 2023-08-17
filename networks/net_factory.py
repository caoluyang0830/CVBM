from networks.BABINet import BABINet
from networks.unet import BABINet2d

def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "BABINet2d":
        net = BABINet2d(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "BABINet2d":
        net = BABINet2d(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "BABINet" and mode == "train":
        net = BABINet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "BABINet" and mode == "test":
        net = BABINet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net
