"""Initialization runs on the authoring host; all model execution is native."""
import math
import numpy as np
from ... import nn


def constant_(parameter,value):
    parameter.assign(np.full(parameter.shape,value,np.float32))
    return parameter


def xavier_uniform_(parameter,gain=1.0):
    if len(parameter.shape)<2: raise ValueError("Xavier initialization requires rank >= 2")
    receptive=math.prod(parameter.shape[2:])
    fan_in,fan_out=parameter.shape[1]*receptive,parameter.shape[0]*receptive
    limit=gain*math.sqrt(6/(fan_in+fan_out))
    parameter.assign(nn._get_rng(None).uniform(-limit,limit,parameter.shape))
    return parameter
