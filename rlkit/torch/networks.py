"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
import math

def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass

    def forward_seq(self,context):
        t,b,_ = context.size()
        input = context.view(t*b,-1)
        out = self.forward(input)
        return out.view(t,b,-1)

class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)


class RNN(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def inner_forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out.contiguous()
        out = out.view(task * seq, -1)

        # output layer
        #preactivation = self.last_fc(out)
        #output = self.output_activation(preactivation)
        if return_preactivations:
            return out, out
        else:
            return out

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        #self.reset(task)
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out [:,-1,:]
        out = out.contiguous()
        out = out.view(task , -1)

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, output
        else:
            return output

    def forward_seq(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        self.reset(task)
        in_=in_.contiguous()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out.contiguous()
        out = out.view(task * seq, -1)

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, output
        else:
            return output

    def inner_reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)

class SnailEncoder(FlattenMlp):
    def __init__(self,
                 input_length,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
        self.input_length = input_length
        # input should be (task, seq, feat) and hidden should be (1, task, feat)

        #self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        layer_count = math.ceil(math.log(input_length)/math.log(2))
        self.TC1 = TCBlock(self.hidden_dim,input_length,16)
        self.atten1 = AttentionBlock(self.hidden_dim+16*layer_count,32,32)
        self.TC2 = TCBlock(self.hidden_dim+16*layer_count+32,input_length,16)
        self.atten2 = AttentionBlock(self.hidden_dim+16*layer_count*2+32,32,32)
        self.out_layer = nn.Linear(self.hidden_dim+16*layer_count*2+32+32,self.output_size)
        self.var_start = int(self.output_size / 2)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out = out.permute(0,2,1)
        #print(out.shape)
        out = self.TC1(out)
        out = self.atten1(out)
        out = self.TC2(out)
        out = self.atten2(out)
        out = out[:, :, -1]
        #print('o',out.shape)
        # output layer
        preactivation = self.out_layer(out)
        output = self.output_activation(preactivation)
        #temp = F.softplus(output[..., self.var_start:])
        #output[..., self.var_start:] = temp
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def forward_seq(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        in_ = in_.contiguous()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out = out.permute(0,2,1)
        #print(out.shape)
        out = self.TC1(out)
        out = self.atten1(out)
        out = self.TC2(out)
        out = self.atten2(out)
        out = out.permute(0,2,1)
        out = out.view(task * seq,-1)


        preactivation = self.out_layer(out)
        output = self.output_activation(preactivation)
        #temp = F.softplus(output[..., self.var_start:])
        #output[..., self.var_start:] = temp
        #output = output.view(task,seq,-1)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self,num_tasks=1):
        return

class MyMlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass

    def forward_seq(self,context):
        t,b,_ = context.size()
        input = context.view(t*b,-1)
        out = self.forward(input)
        return out

    def forward(self,context):
        t,b,_ = context.size()
        input = context.view(t*b,-1)
        out = self.forward(input)
        return out

class CausalConv1d(nn.Module):
    """A 1D causal convolution layer.

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions per step, and T is the number of steps.
    Output: (B, D_out, T), where B is the minibatch size, D_out is the number
        of dimensions in the output, and T is the number of steps.

    Arguments:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = dilation
        self.causal_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2,
            padding = self.padding,
            dilation = dilation
        )

    def forward(self, minibatch):
        return self.causal_conv(minibatch)[:, :, :-self.padding]


class DenseBlock(nn.Module):
    """Two parallel 1D causal convolution layers w/tanh and sigmoid activations

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.

    Arguments:
        in_channels (int): number of input channels
        filters (int): number of filters per channel
    """
    def __init__(self, in_channels, filters, dilation=1):
        super(DenseBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(
            in_channels,
            filters,
            dilation=dilation
        )
        self.causal_conv2 = CausalConv1d(
            in_channels,
            filters,
            dilation=dilation
        )

    def forward(self, minibatch):
        tanh = F.tanh(self.causal_conv1(minibatch))
        sig = F.sigmoid(self.causal_conv2(minibatch))
        out = torch.cat([minibatch, tanh*sig], dim=1)
        return out


class TCBlock(nn.Module):
    """A stack of DenseBlocks which dilates to desired sequence length

    The TCBlock adds `ceil(log_2(seq_len))*filters` channels to the output.

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.

    Arguments:
        in_channels (int): channels for the input
        seq_len (int): length of the sequence. The number of denseblock layers
            is log base 2 of `seq_len`.
        filters (int): number of filters per channel
    """
    def __init__(self, in_channels, seq_len, filters):
        super(TCBlock, self).__init__()
        layer_count = math.ceil(math.log(seq_len)/math.log(2))
        blocks = []
        channel_count = in_channels
        for layer in range(layer_count):
            block = DenseBlock(channel_count, filters, dilation=2**layer)
            blocks.append(block)
            channel_count += filters
        self.blocks = nn.Sequential(*blocks)

    def forward(self, minibatch):
        return self.blocks(minibatch)


class AttentionBlock(nn.Module):
    """An attention mechanism similar to Vaswani et al (2017)

    The input of the AttentionBlock is `BxDxT` where `B` is the input
    minibatch size, `D` is the dimensions of each feature, `T` is the length of
    the sequence.

    The output of the AttentionBlock is `Bx(D+V)xT` where `V` is the size of the
    attention values.

    Arguments:
        input_dims (int): the number of dimensions (or channels) of each element
            in the input sequence
        k_size (int): the size of the attention keys
        v_size (int): the size of the attention values
    """
    def __init__(self, input_dims, k_size, v_size):
        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(input_dims, k_size)
        self.query_layer = nn.Linear(input_dims, k_size)
        self.value_layer = nn.Linear(input_dims, v_size)
        self.sqrt_k = math.sqrt(k_size)

    def forward(self, minibatch):
        minibatch = minibatch.permute(0,2,1)
        keys = self.key_layer(minibatch)
        queries = self.query_layer(minibatch)
        values = self.value_layer(minibatch)
        logits = torch.bmm(queries, keys.transpose(2,1))
        mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).byte()
        mask = torch.triu(mask, 1)
        mask = mask.unsqueeze(0).expand_as(logits)
        logits.data.masked_fill_(mask, float('-inf'))
        probs = F.softmax(logits / self.sqrt_k, dim=2)
        read = torch.bmm(probs, values)
        return torch.cat([minibatch, read], dim=2).permute(0,2,1)