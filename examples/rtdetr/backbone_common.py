from munet import nn
from .utils import get_activation


class ConvNormLayer(nn.Module):
    def __init__(self,ch_in,ch_out,filter_size,stride,groups=1,padding=None,act=None):
        self.conv=nn.Conv2d(ch_in,ch_out,filter_size,stride,padding=(filter_size-1)//2 if padding is None else padding,groups=groups,bias=False)
        self.norm=nn.BatchNorm2d(ch_out)
        self.act=nn.Identity() if act is None else get_activation(act)
    def forward(self,x): return self.act(self.norm(self.conv(x)))
