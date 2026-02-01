""" 
Helper functions for setting up model training & evaluation
"""

import segmentation_models_pytorch as smp
from src.algorithm.models.convlstmseg import ConvLSTMSeg



def get_model(config):

    if config['model']['model_name'] == 'convlstmseg':
        model = ConvLSTMSeg(config=config)
    elif config['model']['model_name'] == 'engelman':
        model = smp.Unet("resnet18", classes=1, in_channels=8, encoder_weights=None)
    else:
        raise ValueError(f"Unknown model: {config['model']['model_name']}")

    return model.to(config['device'])