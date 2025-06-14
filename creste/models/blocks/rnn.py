import torch
import torch.nn as nn

import creste.models.blocks.convgru as convgru
import creste.models.blocks.conv as conv


class MergeUnit(nn.Module):
    def __init__(self,
            input_channels,
            rnn_input_channels=None,
            rnn_config=None,
            costmap_pose_name=None):
        super(MergeUnit, self).__init__()

        if rnn_input_channels is None:
            self.pre_rnn_conv = None
            rnn_input_channels = input_channels
        else:
            self.pre_rnn_conv = conv.ConvLayer(input_channels,
                                                    rnn_input_channels,
                                                    kernel=1,
                                                    bn=True)

        self.costmap_pose_name = costmap_pose_name
        if rnn_config is None:
            self.rnn = None
        else:
            self.force_bos = rnn_config.get('force_bos', False)
            self.groups = rnn_config.get('groups', 1)
            hidden_dims = rnn_config['hidden_dims']

            if rnn_input_channels % self.groups:
                raise Exception(f'RNN input channels {rnn_input_channels}'
                                 ' is not divisible by groups!')
            if any([d % self.groups for d in hidden_dims]):
                raise Exception(f'Not all the hidden_dims are divisible by groups!')


            rnn_input_channels //= self.groups
            hidden_dims = [h//self.groups for h in hidden_dims]

            self.rnn = convgru.ConvGRU(input_size=rnn_config['input_size'],
                                       input_dim=rnn_input_channels,
                                       hidden_dim=hidden_dims,
                                       kernel_size=rnn_config.get('kernel_size', (1,1)),
                                       num_layers=len(hidden_dims),
                                       dtype=torch.cuda.FloatTensor,
                                       batch_first=True,
                                       bias=True,
                                       return_all_layers=True,
                                       noisy_pose=rnn_config.get('noisy_pose', False),
                                       cell_type=rnn_config.get('cell_type', 'GRU'),
                                       align_features=rnn_config.get('align_features', False),
                                       use_z=rnn_config.get('use_z', False),
            )

    # Here bos means beginning of sequence
    def forward(self, x, t=1, bos=None, pose=None):
        """
        Inputs:
            x - [B, T, C, H, W] image tensor
            t - number of frames in the sequence
            bos - [BT] boolean tensor indicating beginning of sequence
        Outputs:
            x - [BT, C, H, W] image tensor
        """
        if self.pre_rnn_conv is not None:
            x = self.pre_rnn_conv(x)

        if self.rnn is not None:
            assert(bos is not None) # and pose is not None) # x is pretransformed
            if self.force_bos:
                bos = torch.ones_like(bos)
                t = 1

            ### reshape (bt, c, h, w) --> (b, t, c, h, w)
            bt, c, h, w = x.shape
            b = bt//t
            bos = bos.reshape(b, t)
            # pose = pose.reshape(b, t, 4, 4)

            if self.groups > 1:
                bg = b * self.groups

                # move groups to batch
                assert(c % self.groups == 0)
                x = x.reshape(b, t, self.groups, c//self.groups, h, w)
                # t <-> self.groups
                x = x.transpose(1, 2)
                x = x.reshape(bg, t, c//self.groups, h , w)

                bos = bos.repeat(self.groups, 1)
                # pose = pose.repeat(self.groups, 1 , 1, 1)
            else:
                x = x.reshape(b, t, c, h, w)

            # We simplify things assuming that bos[:, t] is *all* True or False
            assert(torch.all(torch.all(bos, axis=0) ^ torch.all(~bos, axis=0)))

            # Also we furthur simplify things assuming that only bos[:, 0] can be true :)
            assert(torch.any(bos[0, 1:]) == False), ('Only the first element in the chunk '
                                                     'can be begging of the sequence. '
                                                     'Make sure "miniseq_sampler.len" is '
                                                     'divisible by "miniseq_sampler.chunk_len".')

            ###### DEBUG: Trun off input for a few frames ###
            #if hasattr(self, 'test'):
            #    self.test += 1
            #else:
            #    self.test = 0

            #if (self.test % 20) > 10:
            #    x = torch.zeros_like(x)
            #################################################

            if bos[0, 0]:  # start of a new sequence
                self.hidden_state = None

            ## We keep the translation and resolution the same for
            # all the rnn layers so we can use same pose for all
            # layers.
            # pose = pose[:, :, None].expand(-1, -1, self.rnn.num_layers, -1, -1)
            layer_output_list, last_state_list = self.rnn(
                x, 
                # pose,                                                                 
                hidden_state=self.hidden_state
            )

            self.hidden_state = []
            for state in last_state_list:
                assert(len(state) == 1)
                dstate = state[0].detach()
                dstate.requires_grad = True
                self.hidden_state.append(dstate) # should be [B x C x H x W]

            x = layer_output_list[-1]

            if self.groups > 1:
                x = x.reshape(b, self.groups, t, c//self.groups, h, w)
                # self.groups <-> t
                x = x.transpose(1, 2)

            x = x.reshape(bt, -1, h, w)

        return x


