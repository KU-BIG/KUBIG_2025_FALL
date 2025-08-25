import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PoolingBaseModel(nn.Module):
    """Inherit from this class when implementing new models."""

    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True, suffix_tensor_name=""):
        super(PoolingBaseModel, self).__init__()
        
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.suffix_tensor_name = suffix_tensor_name

        self.gating_weights = nn.Parameter(torch.randn(output_dim, output_dim) / math.sqrt(output_dim))

        if self.add_batch_norm:
            self.gating_bn = nn.BatchNorm1d(output_dim)
        else:
            self.gating_biases = nn.Parameter(torch.randn(output_dim) / math.sqrt(output_dim))

    def context_gating(self, input_layer):
        """Context Gating in PyTorch.

        Args:
        input_layer: Input layer in the following shape:
        'batch_size' x 'number_of_activation'

        Returns:
        activation: gated layer in the following shape:
        'batch_size' x 'number_of_activation'
        """
        gates = F.linear(input_layer, self.gating_weights.T) # PyTorch linear layer performs matmul with transpose of weights
        
        if self.add_batch_norm:
            gates = self.gating_bn(gates)
        else:
            gates += self.gating_biases
        
        gates = torch.sigmoid(gates)
        activation = input_layer * gates
        return activation

    def forward(self, reshaped_input):
        raise NotImplementedError("Models should implement the forward pass.")

# -----------------------------------------------------------------------------

class NetRVLAD(PoolingBaseModel):
    """Creates a NetRVLAD class (Residual-less NetVLAD)."""
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True, suffix_tensor_name=""):
        super(NetRVLAD, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training,
            suffix_tensor_name=suffix_tensor_name)
        
        self.cluster_weights = nn.Parameter(torch.randn(self.feature_size, self.cluster_size) / math.sqrt(self.feature_size))
        
        if self.add_batch_norm:
            self.cluster_bn = nn.BatchNorm1d(self.cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(self.cluster_size) / math.sqrt(self.feature_size))
            
        self.hidden1_weights = nn.Parameter(torch.randn(self.cluster_size * self.feature_size, self.output_dim) / math.sqrt(self.cluster_size))
            
    def forward(self, reshaped_input):
        """Forward pass of a NetRVLAD block."""
        
        activation = F.linear(reshaped_input, self.cluster_weights.T)

        if self.add_batch_norm:
            activation = self.cluster_bn(activation)
        else:
            activation += self.cluster_biases

        activation = F.softmax(activation, dim=-1)

        activation = activation.view(-1, self.max_samples, self.cluster_size)
        activation = activation.permute(0, 2, 1)

        reshaped_input = reshaped_input.view(-1, self.max_samples, self.feature_size)

        vlad = torch.bmm(activation, reshaped_input)
        
        vlad = vlad.permute(0, 2, 1)
        vlad = F.normalize(vlad, p=2, dim=1)

        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad, p=2, dim=1)

        vlad = F.linear(vlad, self.hidden1_weights.T)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad

# -----------------------------------------------------------------------------

class NetVLAD(PoolingBaseModel):
    """Creates a NetVLAD class."""
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True, suffix_tensor_name=""):
        super(NetVLAD, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training,
            suffix_tensor_name=suffix_tensor_name)
            
        self.cluster_weights = nn.Parameter(torch.randn(self.feature_size, self.cluster_size) / math.sqrt(self.feature_size))
        
        if self.add_batch_norm:
            self.cluster_bn = nn.BatchNorm1d(self.cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(self.cluster_size) / math.sqrt(self.feature_size))
            
        self.cluster_weights2 = nn.Parameter(torch.randn(1, self.feature_size, self.cluster_size) / math.sqrt(self.feature_size))
        
        self.hidden1_weights = nn.Parameter(torch.randn(self.cluster_size * self.feature_size, self.output_dim) / math.sqrt(self.cluster_size))
            
    def forward(self, reshaped_input):
        """Forward pass of a NetVLAD block."""

        activation = F.linear(reshaped_input, self.cluster_weights.T)

        if self.add_batch_norm:
            activation = self.cluster_bn(activation)
        else:
            activation += self.cluster_biases
        
        activation = F.softmax(activation, dim=-1)

        activation_reshaped = activation.view(-1, self.max_samples, self.cluster_size)
        a_sum = torch.sum(activation_reshaped, dim=-2, keepdim=True)
        a = a_sum * self.cluster_weights2
        a = a.permute(0, 2, 1)

        activation = activation_reshaped.permute(0, 2, 1)
        
        reshaped_input = reshaped_input.view(-1, self.max_samples, self.feature_size)
        
        vlad = torch.bmm(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1)
        vlad = vlad - a
        
        vlad = F.normalize(vlad, p=2, dim=1)

        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad, p=2, dim=1)
        
        vlad = F.linear(vlad, self.hidden1_weights.T)
        
        if self.gating:
            vlad = self.context_gating(vlad)
            
        return vlad