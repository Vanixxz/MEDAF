from .train import *
from .test import *
from .net import BaselineNet, MultiBranchNet

def get_model(args):
    net = MultiBranchNet(args)
    return net
