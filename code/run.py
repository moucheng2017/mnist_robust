import argparse
from train import *


def main():
    parser = argparse.ArgumentParser('training on kannada mnist', add_help=False)
    parser.add_argument('--seed', '-s', default=1024, type=int, help='Random seed')
    parser.add_argument('--net', default='moe', type=str, help='resent or vgg or simplenet')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--temp', default=2.0, type=float, help='Temperature')
    parser.add_argument('--width', default=10, type=int, help='Network width')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--loss_fun', default='ce', type=str, help='loss function')
    parser.add_argument('--lr_decay', default=0, type=int, help='1 when decays the lr or 0 without decaying')
    parser.add_argument('--batch', default=1024, type=int, help='batch size of labelled data for training')
    parser.add_argument('--epochs', default=20, type=int, help='total training steps')

    parser.add_argument('--train_noise', default=0, type=int, help='type of training noises')
    parser.add_argument('--test_noise', default=1, type=int, help='type of test noises')

    parser.add_argument('--epsilon', default=5, type=float, help='strength of attack')

    global args
    args = parser.parse_args()
    trainer(args)


if __name__ == '__main__':
    main()