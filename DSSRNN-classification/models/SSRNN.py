import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Layers for input and hidden transformations
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden):
        # Compute transformations for input and hidden state
        input_transformation = self.input_layer(x)
        hidden_transformation = self.hidden_layer(hidden)

        # Apply non-linearity (tanh is commonly used in RNNs)
        activation = torch.relu(input_transformation + hidden_transformation)

        # Sum activation output with the current state to get the next state
        # hidden_next = activation + hidden
        out = torch.relu(activation + hidden_transformation)
        return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_size = configs.hidden_size
        self.channels = configs.enc_in
        self.num_layers = configs.num_layers
        self.individual = configs.individual
        self.batch_size = configs.batch_size

        if self.individual:
            self.rnn_cells = nn.ModuleList([
                CustomRNNCell(input_size=1, hidden_size=self.hidden_size)
                for _ in range(self.channels)
            ])
        else:
            # self.rnn_cell = CustomRNNCell(input_size=self.channels, hidden_size=self.hidden_size)
            self.rnn_cell = CustomRNNCell(input_size=self.seq_len, hidden_size=self.hidden_size)

        # self.fc = nn.Linear(self.hidden_size, self.channels)
        # self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.fc = nn.Linear(self.hidden_size, self.pred_len)
            

    def forward(self, x, batch_x_mark=None, dec_inp=None, batch_y_mark=None, batch_y=None):
        # x.shape: torch.Size([16, 96, 21])
        # [Batch, Input length, Channel]
        # print(f'x.shape------------------------------: {x.shape}')
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)

        if self.individual:
            outputs = []
            for i in range(self.channels):
                x_channel = x[:, :, i:i+1]  # Shape: [Batch, Input length, 1]
                h = torch.zeros(batch_size, self.hidden_size).to(x.device)
                out = []
                for t in range(x_channel.size(1)):
                    h = self.rnn_cells[i](x_channel[:, t], h)
                    out.append(h.unsqueeze(1))
                out = torch.cat(out, dim=1)
                outputs.append(out)
            out = torch.cat(outputs, dim=-1)
        else:
            h = torch.zeros(x.size(1), self.hidden_size).to(x.device)
            
            out = []
            for t in range(x.size(0)):
                h = self.rnn_cell(x[t, :, :].squeeze(), h)
                # print(f'h.shape: {h.shape}')
                # print(f'x[:, t, :].squeeze().shape: {x[t, :, :].squeeze().shape}')
                out.append(h.unsqueeze(1))
            out = torch.cat(out, dim=1)

        # Reshape the output to [Batch, Output length, Channel]
        out = self.fc(out)

        # out = out.view(batch_size, self.seq_len, -1)
        # out = out[:, -self.pred_len:, :]
        # print("after")
        # print(out.shape)
        # out = self.Linear(x).permute(0,2,1)
        # print("outputofmodel")
        # print(out)
        out = torch.sigmoid(out)
        # out = torch.round(out)



        threshold = 0.5

        out = torch.where(out >= threshold, torch.tensor(1.0), torch.tensor(0.0))
        out = out.requires_grad_(True)  


        # print("outputofmodel")
        # print(out)


        out = out.permute(1, 2, 0)
        # out = out[:, -self.pred_len:, :]
        # print("after permute")
        # print(out.shape)
        # out = torch.softmax(out, axis=2)
        # print("softm-----------")
        # print(out)
        # threshold = 0.4

        # out = torch.where(out >= threshold, torch.tensor(1.0), torch.tensor(0.0))
        # out = out.requires_grad_(True)  
        
        # out = torch.where(out > torch.tensor(0.7), torch.tensor(1), torch.tensor(0))
        # out = torch.where(out > torch.tensor(0.7, dtype=torch.float32), torch.tensor(1), torch.tensor(0))
        # print("round-----------")
        # print(out)

        return out
