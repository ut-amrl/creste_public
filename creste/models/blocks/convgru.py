'''
Original code:
	https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
'''

import os
import torch
from torch import nn
from torch.autograd import Variable
from creste.utils.utils import warp
# from bevnet.models.feature_alignment import FeatureAlignment

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype, cell_type):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param cell_type: either 'GRU' or 'MRU'
        """
        super(ConvGRUCell, self).__init__()

        # see https://arxiv.org/pdf/1603.09420.pdf for MRU
        self.cell_type = cell_type
        assert(cell_type in ['GRU', 'MRU'])

        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        gate_out = hidden_dim + hidden_dim * (cell_type == 'GRU')

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=gate_out,
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim,
                                     self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        if self.cell_type == 'GRU':
            gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
            reset_gate = torch.sigmoid(gamma)
            update_gate = torch.sigmoid(beta)
        else: # mru
            reset_gate = torch.sigmoid(combined_conv)
            update_gate = reset_gate
        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRUCellSimple(nn.Module):
    """
    A Simple cell that doesn't do any convolution.
    """
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype, cell_type):
        super(ConvGRUCellSimple, self).__init__()
        assert(cell_type == 'simple')
        self.height, self.width = input_size
        assert input_dim == hidden_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim,
                                     self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        return input_tensor + h_cur


class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False,
                 use_pose=False, noisy_pose=False, cell_type='GRU',
                 align_features=False, use_z=False,
                 warp_precision=torch.float64):
        """

        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.use_pose = use_pose
        self.noisy_pose = noisy_pose
        self.cell_type = cell_type
        self.use_z = use_z
        self.warp_precision = warp_precision

        if use_z:
            assert(len(hidden_dim) == 1)
            self.z_map = nn.Sequential(
                   nn.Linear(1, hidden_dim[0], bias=True),
                   nn.ReLU(),
                   nn.Linear(hidden_dim[0], hidden_dim[0], bias=True)
            )

        cell_list = []
        align = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]

            if cell_type in ['GRU', 'MRU']:
                cell_class = ConvGRUCell
            elif cell_type == 'simple':
                cell_class = ConvGRUCellSimple
            else:
                raise ValueError('Unknown cell type', cell_type)

            cell_list.append(cell_class(input_size=(self.height, self.width),
                                        input_dim=cur_input_dim,
                                        hidden_dim=self.hidden_dim[i],
                                        kernel_size=self.kernel_size[i],
                                        bias=self.bias,
                                        dtype=self.dtype,
                                        cell_type=self.cell_type))

            if align_features:
                align.append(FeatureAlignment(cur_input_dim))

        # convert python list to pytorch module
        if align:
            self.align = nn.ModuleList(align)
        else:
            self.align = None

        self.cell_list = nn.ModuleList(cell_list)


    def _noisify(self, delta_pose, rotation_noise_scale=0.01, translation_noise_scale=0.1):
        """
        Args:
            delta_pose: A B x 2 x 3 affine matrices

        Returns:
            A noisified version of the input
        """
        batch_size = delta_pose.size(0)
        rotation_noise = torch.randn(batch_size, device=delta_pose.device) * rotation_noise_scale
        translation_noise = torch.randn(batch_size, 2, device=delta_pose.device) * translation_noise_scale
        s = torch.sin(rotation_noise)
        c = torch.cos(rotation_noise)
        R = delta_pose.new_zeros((batch_size, 2, 2))
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        out = delta_pose.new_zeros((batch_size, 2, 3))
        out[:, :2, :2] = torch.matmul(R, delta_pose[:, :2, :2])
        out[:, :2, 2] = delta_pose[:, :2, 2] + translation_noise
        return out

    def forward(self, input_tensor, pose=None, hidden_state=None):
        """

        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param pose: (b, t, num_layers, 3, 3)
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            pose = pose.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            pass
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        if self.use_pose:
            assert(pose is not None)

        layer_output_list = []
        last_state_list   = []
        last_cell_poses = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            if self.use_pose:
                cell_pose = hidden_state[self.num_layers + layer_idx]

            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state
                # then compute the next hidden and
                # cell state through ConvLSTMCell forward function
                if self.use_pose:
                    input_pose = pose[:, t, layer_idx]

                    ## transform h to input_pose coordinate frame
                    if cell_pose is not None:
                        # cell_pose -> input_pose transformation
                        # inv(input_pose) * cell_pose

                        def _2d(pose3d):
                            return pose3d[:, (0, 1, 3)][:, :, (0, 1, 3)]

                        M = torch.matmul(
                                torch.inverse(_2d(input_pose)),
                                _2d(cell_pose))[:, :2]

                        if self.noisy_pose:
                            M = self._noisify(M)

                        transformed_h, _ = warp(h, M, interpolation='bilinear',
                                                precision=self.warp_precision)


                        if self.use_z:
                            dz = -input_pose[:, 2, 3] + cell_pose[:, 2, 3]
                            mapped_z = self.z_map(dz[:, None])
                            transformed_h = transformed_h + torch.tanh(mapped_z[..., None, None])

                        ## Debugging
                        #from matplotlib.pyplot import show, imshow, figure, imsave, title
                        #for b in range(len(h)):
                        #    features = [cur_layer_input[b, t, :, :, :],
                        #                transformed_h[b], h[b]]
                        #    names = ['0input', '1corrected_h', '2h']
                        #    for i, fea in enumerate(features):
                        #        img = fea.cpu().detach().numpy().mean(axis=0)
                        #        name = f'temp/{b}_{t}_{names[i]}.png'
                        #        figure()
                        #        imshow(img)
                        #        title(name)
                        #        imsave(name, img)
                        #show()
                        #from IPython import embed;embed()
                        ################

                        h = transformed_h

                # (b,t,c,h,w)
                inp = cur_layer_input[:, t, :, :, :]

                if self.align is not None:
                    h = self.align[layer_idx](h, inp)

                h = self.cell_list[layer_idx](
                    input_tensor=inp,
                    h_cur=h)

                output_inner.append(h)

                if self.use_pose:
                    cell_pose = input_pose

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

            if self.use_pose:
                last_cell_poses.append([cell_pose])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]
            last_cell_poses = last_cell_poses[-1:]

        return layer_output_list, (last_state_list + last_cell_poses)

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))

        if self.use_pose:
            for i in range(self.num_layers):
                init_states.append(None)

        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor # computation in GPU
    else:
        dtype = torch.FloatTensor

    height = width = 6
    channels = 256
    hidden_dim = [32, 64]
    kernel_size = (3,3) # kernel size for two stacked hidden layer
    num_layers = 2 # number of stacked hidden layer
    model = ConvGRU(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    dtype=dtype,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)

    batch_size = 1
    time_steps = 5
    input_tensor = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w)

    if use_gpu:
        input_tensor = input_tensor.cuda()
        model.cuda()
    layer_output_list, last_state_list = model(input_tensor)
