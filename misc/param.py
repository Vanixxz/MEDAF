import os
import yaml
import argparse

from easydict import EasyDict as edict


parser = argparse.ArgumentParser(description='PyTorch Implementation')


def parser2dict():
    config, _ = parser.parse_known_args()
    option = edict(config.__dict__)
    return edict(option)


def _merge_a_into_b(a, b):
    if type(a) is not edict:
        return
    for k, v in a.items():
        if type(v) is edict:
            _merge_a_into_b(a[k], b)
        else:
            if k in str(b.items()):
                continue
            else:
                b[k] = v


def print_options(args):
    message = ''
    message += ' < Options > \n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:<15}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += ' <  End  >\n'
    print(message)


def get_config(task):
    option = parser2dict()
    if 'POSE_PARAM_PATH' in os.environ:
        filename = os.environ['POSE_PARAM_PATH'] + '/misc/' + task + '.yml'
    else:
        filename = 'misc/' + task + '.yml'
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))
    _merge_a_into_b(yaml_cfg, option)

    return option


parser.add_argument('-c', '--ckpt', default='', type=str)
parser.add_argument('-r', '--resume', action='store_true')
parser.add_argument('-g', '--gpu_ids', type=str, default='0')
parser.add_argument('-p', '--plus_num', type=int, default=10)
parser.add_argument('-d', '--dataset', type=str, default='tiny_imagenet')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N')
parser.add_argument('--seed', default=71324, type=int)