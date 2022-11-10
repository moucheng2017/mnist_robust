import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn


def preprocess(train_noise=0, test_noise=0):

    if train_noise == 0:
        train_x = np.load('../data/np/train/raw.npy')
    else:
        train_x = np.load('../data/np/train/gaussian2.npy')

    train_y = np.load('../data/np/train/labels.npy')

    if test_noise == 0:
        test_x = np.load('../data/np/test/raw.npy')
    elif test_noise == 1:
        test_x = np.load('../data/np/test/gaussian5.npy')
    elif test_noise == 2:
        test_x = np.load('../data/np/test/gaussian10.npy')
    elif test_noise == 3:
        test_x = np.load('../data/np/test/blurred5.npy')
    elif test_noise == 4:
        test_x = np.load('../data/np/test/blurred7.npy')
    else:
        test_x = np.load('../data/np/test/raw.npy')
        # with gradient attack

    test_y = np.load('../data/np/test/labels.npy')

    return train_x, train_y, test_x, test_y


def get_dataloaders(train_x, train_y, test_x, test_y, batch):

    torch_train_x = torch.from_numpy(train_x).type(torch.FloatTensor).to('cuda')
    torch_train_y = torch.from_numpy(train_y).type(torch.FloatTensor).to('cuda')

    train_dataset = torch.utils.data.TensorDataset(torch_train_x, torch_train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)

    torch_test_x = torch.from_numpy(test_x).type(torch.FloatTensor).to('cuda')
    torch_test_y = torch.from_numpy(test_y).type(torch.FloatTensor).to('cuda')

    test_dataset = torch.utils.data.TensorDataset(torch_test_x, torch_test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False, drop_last=False)

    return train_loader, test_loader


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]

        dc = dc.mean()

        return 1-dc


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp
