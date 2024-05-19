# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the linear model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
import torch.nn.functional as F

#########
# MODEL #
#########

def get_downstream_model(input_dim, output_dim, config):
    model_cls = eval(config['select'])
    model_conf = config.get(config['select'], {})
    model = model_cls(input_dim, output_dim, **model_conf)
    return model

class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_state, features_len=None):
        logit = self.linear(hidden_state)

        return logit, features_len

class UtteranceLevel(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        pooling='MeanPooling',
        activation='ReLU',
        post_net={'select': 'Linear'},
        **kwargs
    ):
        super().__init__()
        latest_dim = input_dim
        self.pooling = eval(pooling)(input_dim=latest_dim, activation=activation)
        self.post_net = get_downstream_model(latest_dim, output_dim, post_net)

    def forward(self, hidden_state, features_len=None):
        pooled, features_len = self.pooling(hidden_state, features_len)
        
        logit, features_len = self.post_net(pooled, features_len)

        return logit, features_len

class MeanPooling(nn.Module):

    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__()

    def forward(self, feature_BxTxH, features_len, **kwargs):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            agg_vec = torch.mean(feature_BxTxH[i][:features_len[i]], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list), torch.ones(len(feature_BxTxH)).long()

