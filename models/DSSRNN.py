import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.SSRNN import Model as SSRNNModel


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition Kernel Size
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)  # Ensure series_decomp is correctly imported
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.SSRNN_Seasonal = nn.ModuleList()
            self.SSRNN_Trend = nn.ModuleList()
            
            for _ in range(self.channels):
                # Create new instances of SSRNNModel for each channel
                self.SSRNN_Seasonal.append(SSRNNModel(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            # Create a single instance of SSRNNModel for both seasonal and trend components
            self.SSRNN_Seasonal = SSRNNModel(configs)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.SSRNN_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.SSRNN_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x, batch_x_mark=None, dec_inp=None, batch_y_mark=None, batch_y=None):
        # x: [Batch, Input length, Channel]
        # print(f'x.shape: {x.shape}')
        seasonal_init, trend_init = self.decomposition(x)
        # print(f'seasonal_init.shape: {seasonal_init.shape}')
        # print(f'trend_init.shape: {trend_init.shape}')
        trend_init = trend_init.permute(0,2,1)
        # print('----------------------------------')
        # print(f'seasonal_init.shape: {seasonal_init.shape}')
        # print(f'trend_init.shape: {trend_init.shape}')
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.SSRNN_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.SSRNN_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.SSRNN_Seasonal(seasonal_init)
            # print(f'seasonal_output.shape: {seasonal_output.shape}')
            trend_output = self.Linear_Trend(trend_init)
            # print(f'trend_output.shape: {trend_output.shape}')
        
        trend_output = trend_output.permute(0,2,1)
        # print(f'trend_output_afterpermute.shape: {trend_output.shape}')
        x = seasonal_output + trend_output
        return x # to [Batch, Output length, Channel]
