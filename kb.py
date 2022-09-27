import random
import os
import pygame
import gsn.gsn as gsn
import nn.ImgNet as nn
import torch
import numpy as np 
import math


class KnowledgeBase:
    def __init__(self, gsn_file, nn_file):
        self.gsn = gsn.GSNModel(gsn_file)
        self.nn = nn._load_model(nn_file)
        