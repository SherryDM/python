# StarGAN简版

import os
import argparse
from mySolver import Solver
from myDataloader import get_loader
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.x_hat_dir):
        os.makedirs(config.x_hat_dir)

    A_loader = get_loader('/data/dm/data/BU_3DFE/train', crop_size=900,
                          image_size=100, batch_size=8, mode='train')
    B_loader = get_loader('/data/dm/data/RAF_DB/train', crop_size=100,
                          image_size=100, batch_size=8, mode='train')


    # Solver for training and testing StarGAN.
    solver = Solver(A_loader, B_loader, config)
    solver.train_multi()
    # solver.test_multi()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=7, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=7, help='dimension of domain labels (2nd dataset)')
    # parser.add_argument('--celeba_crop_size', type=int, default=100, help='crop size for the CelebA dataset')##
    # parser.add_argument('--rafd_crop_size', type=int, default=300, help='crop size for the RaFD dataset')#裁剪rafd人脸
    parser.add_argument('--image_size', type=int, default=100, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=5, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=5, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=20000, help='number of total iterations for training D')##训练迭代次数
    parser.add_argument('--num_iters_decay', type=int, default=10000, help='number of iterations for decaying lr')##
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')##调参学习率G
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')##调参学习率D
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')##调参
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')##调参
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['angry', 'disgusted', 'fearful', 'happy', 'neutral','sad','suprised'])##更改源域属性标签

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=20000, help='test model from this step')###

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    # parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    # parser.add_argument('--celeba_image_dir', type=str, default='/data/dm/data/RAF_DB/aligned')
    # parser.add_argument('--attr_path', type=str, default='/data/dm/data/RAF_DB/all_alignedlabel.txt')
    # parser.add_argument('--rafd_image_dir', type=str, default='/data/dm/data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')
    parser.add_argument('--x_hat_dir', type=str, default='stargan/x_hat')

    # Step size.
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=100)

    config = parser.parse_args()
    print(config)
    main(config)