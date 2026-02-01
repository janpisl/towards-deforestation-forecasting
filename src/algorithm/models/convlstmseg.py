import torch.nn as nn
import torch
import pdb

from src.algorithm.models.model_blocks import ConvLSTM


class ConvLSTMSeg(nn.Module):
    """Wrapper of ConvLSTM followed by a 1D Conv layer to produce an output map
    """
    def __init__(self, config):
        super(ConvLSTMSeg, self).__init__()

        input_channels = 0
        if 'deforestation_map' in config['inputs']:
            input_channels += 1
        if 'elevation' in config['inputs']:
            input_channels += 1
        if 'distance' in config['inputs']:
            input_channels += 1
        if 'disturbances' in config['inputs']:
            input_channels += 1
        if 'lulc' in config['inputs']:
            input_channels += config['model']['params']['lulc_out_channels']
            self.lulc_mapping = nn.Sequential(
                nn.Conv2d(
                    in_channels=38, #This is the number of classes in MapBiomas LULC
                    out_channels=config['model']['params']['lulc_out_channels'],
                    kernel_size=1,
                    #bias=False
                    ),
                #nn.ReLU(inplace=True)
            )
        if 'protected' in config['inputs']:
            input_channels += 1
            self.protected_areas_mapping = nn.Sequential(
                nn.Conv2d(
                    in_channels=4, #One-hot encoding of 4 values
                    out_channels=1,
                    kernel_size=1,
                    ),
            )
 

        hidden_dim = config['model']['params']['hidden_dim']
        kernel_size = tuple(config['model']['params']['kernel_size'])
        num_layers = config['model']['params']['num_layers']
        return_all_layers = config['model']['params']['return_all_layers']
        activation = config['model']['params']['activation']

        self.convlstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=return_all_layers,
            activation=activation)

        conv_kernel_size = tuple(config['model']['params']['conv_kernel_size'])

        self.conv = nn.Conv2d(
            in_channels=hidden_dim[-1],
            out_channels=1,
            kernel_size=conv_kernel_size
            )

    
    def forward(self, input_dict):

        if 'lulc' in input_dict.keys():
            B, T, C, H, W = input_dict['lulc'].shape
            input_dict['lulc'] = self.lulc_mapping(input_dict['lulc'].contiguous().view(B * T, C, H, W)).view(B, T, -1, H, W)

        
        if 'protected' in input_dict.keys():
            #Map one-hot encoding to a feature vector
            input_dict['protected'] = self.protected_areas_mapping(input_dict['protected'].squeeze(1))
            
        timesteps = input_dict['deforestation_map'].shape[1]

        # Repeat non-temporal inputs
        for key in ['elevation', 'protected']:
            try:
                input_dict[key] = input_dict[key].unsqueeze(1).expand(-1, timesteps, -1, -1, -1)
            except KeyError:
                pass
        
        input_tensor = torch.cat([value for _, value in input_dict.items()], dim=2)

        layer_output_list, last_state_list = self.convlstm(input_tensor)
        
        #layer_output_list is a list with one item per layer, each item.shape B,T,C,W,H
        #so i concatenate it along the channel dimension 
        #feature_maps = torch.cat(layer_output_list, dim=2)
        feature_maps = layer_output_list[-1]

        B, T, C, H, W = feature_maps.shape
        x = feature_maps.contiguous().view(B * T, C, H, W)
        x = self.conv(x)
        outputs = x.view(B, T, *x.shape[1:])

        return outputs
