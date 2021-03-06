import os
import argparse
import numpy as np
import torch

import wandb

from core.data_provider import datasets_factory
from core.data_provider.weatherDataset import get_true_paths_from_csv
from core.models.model_factory import Model
import core.trainer as trainer
# import pynvml
import torchio as tio
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# pynvml.nvmlInit()
# -----------------------------------------------------------------------------
# from tensor_board import Tensorboard


def schedule_sampling(eta, itr, channel, batch_size,args):
    zeros = np.zeros((batch_size,
                      args.total_length - args.input_length,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    # print('eta: ', eta)
    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length ))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - args.input_length ):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - args.input_length ,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * channel))
    return eta, real_input_flag


def train_wrapper(model,args,key,wandb):
    begin = 0
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # meminfo_begin = pynvml.nvmlDeviceGetMemoryInfo(handle)

    if args.pretrained_model:
        model.load(args.pretrained_model)
        begin = int(args.pretrained_model.split('-')[-1])
    data_path = os.path.join("data", "Train")
    csv_path = os.path.join(data_path, "train.csv")
    data_paths = get_true_paths_from_csv(data_path, csv_path)
    data_paths = data_paths[key]
    train_data_paths = data_paths[0:int(len(data_paths)*0.9)]
    valid_data_paths = data_paths[int(len(data_paths)*0.9):]
    print("train data length {}".format(len(train_data_paths)),"valid data length {}".format(len(valid_data_paths)))
    target_size = (args.total_length, args.img_height, args.img_width)

    resize =    tio.Resize(target_shape=target_size)
    trainTransform = tio.Compose([
        resize,
    ])
    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        data_train_path=train_data_paths,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True,
                                                        train_transform=trainTransform,
                                                        key = key)
    val_input_handle = datasets_factory.data_provider(configs=args,
                                                      data_train_path=valid_data_paths,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False,
                                                      train_transform=trainTransform,
                                                      key = key)
    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itr = begin
    avg_ssim = 0
    for epoch in range(0, args.max_epoches):
        if itr > args.max_iterations:
            break
        for ims in train_input_handle:
            total_itr= len(train_input_handle)
            if itr > args.max_iterations:
                break
            batch_size = ims.shape[0]
            eta, real_input_flag = schedule_sampling(eta, itr, args.img_channel, batch_size,args)
            loss_l1, loss_l2=0,0
            if itr % args.test_interval == 0:
                print('Validate:')
                cur_avg_ssim = trainer.test(model, val_input_handle, args, itr,key,wandb)
                if cur_avg_ssim > avg_ssim:
                    print("model saved")
                    model.save(itr,key)
            loss_l1, loss_l2 = trainer.train(model, ims, real_input_flag, args, itr,total_itr)
            if itr % args.display_interval == 0:
                info = {"{}_train_loss_l1".format(key): loss_l1,
                        "{}_train_l2".format(key): loss_l2,
                        }

                wandb.upload_wandb_info(info_dict=info)
            # if itr % args.snapshot_interval == 0 and itr > begin:
            #     model.save(itr,key)
            itr += 1

            # meminfo_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("GPU memory:%dM" % ((meminfo_end.used - meminfo_begin.used) / (1024 ** 2)))


def test_wrapper(model,args,key,wandb):
    # model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       data_train_path=args.data_train_path,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False,
                                                       key=key)

    itr = 1
    for i in range(itr):
        trainer.test(model, test_input_handle, args, itr,key,wandb)
def main():
    parser = argparse.ArgumentParser(description='MAU')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--is_train', type=str, default='False', required=True)
    parser.add_argument('--max_iterations', type=int, default=10000)

    args_main = parser.parse_args()
    args_main.tied = True

    if args_main.is_train == 'True':
        # from configs.mnist_train_configs import configs
        from configs.weather_train_configs import configs

    else:
        from configs.mnist_configs import configs

    parser = configs()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    args.tied = True
    os.environ['WANDB_API_KEY'] = "55a895793519c48a6e64054c9b396629d3e41d10"
    args.use_wandb = True
    args.project_name = "TianchWeather"
    args.method_name = "MAU"
    wandb = Tensorboard(args)
    print('Initializing models')
    if args.is_training == 'True':
        args.is_training = True
    else:
        args.is_training = False
    # args.batch_size=78
    model = Model(args)
    keys = ["radar","precip","wind"]
    for key in keys:
        if args.is_training:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            if not os.path.exists(args.gen_frm_dir):
                os.makedirs(args.gen_frm_dir)
            # train_wrapper(model,args,key)
            train_wrapper(model=model,args=args,key=key,wandb=wandb)

        else:
            if not os.path.exists(args.gen_frm_dir):
                os.makedirs(args.gen_frm_dir)
            # test_wrapper(model,args,key,wandb)
            test_wrapper(model=model,args=args,key=key,wandb=wandb)


class Tensorboard:
    def __init__(self, config):
        os.system("wandb login")
        os.system("wandb {}".format("online" if config.use_wandb else "offline"))
        config.run_name = "MAU Num_hidden {} Num_layers {} TAU {}".format(str(config.num_hidden),
                                                                          str(config.num_layers),
                                                                          str(config.tau))
        self.tensor_board = wandb.init(project=config.project_name,
                                       name=config.run_name,
                                       config=config)
        self.ckpt_root = 'saved'
        self.ckpt_path = os.path.join(self.ckpt_root, config.run_name)
        self.visual_root_path = os.path.join(self.ckpt_path, 'history_images')
        self.visual_results_root = os.path.join(self.visual_root_path, 'results')
        self._safe_mkdir(self.ckpt_root)
        self._safe_mkdir(self.ckpt_path)
        self._safe_mkdir(self.visual_root_path)
        self._safe_mkdir(self.visual_results_root)
        self._safe_mkdir(self.ckpt_root, config.run_name)

    def upload_wandb_info(self, info_dict, current_step=0):
        for i, info in enumerate(info_dict):
            self.tensor_board.log({info: info_dict[info]})
        return

    @staticmethod
    def _safe_mkdir(parent_path, build_path=None):
        if build_path is None:
            if not os.path.exists(parent_path):
                os.mkdir(parent_path)
        else:
            if not os.path.exists(os.path.join(parent_path, build_path)):
                os.mkdir(os.path.join(parent_path, build_path))
        return

    def save_ckpt(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.ckpt_path, name))
        return


if __name__ == '__main__':

    main()
