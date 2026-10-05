""" Encoder-decoder architecture based on ERFNet
"""
import torch
import torch.nn as nn
from . import erfnet

class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None, batch_norm=True, instance_norm=False):
        super().__init__()

        print('Creating branched erfnet with {} classes'.format(num_classes))
        assert not(batch_norm & instance_norm)

        if (encoder is None):
            self.encoder = erfnet.Encoder(sum(num_classes), batch_norm, instance_norm)
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n, batch_norm, instance_norm))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            # --- objects ---
            # object offsets (dx, dy)
            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            if n_sigma <= 2:
                # object sigmas (sx, sy)
                output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
                output_conv.bias[2:2+n_sigma].fill_(0.75)
            else:
                # object sigmas (sx, sy)
                output_conv.weight[:, 2:2+2, :, :].fill_(0)
                output_conv.bias[2:2+2].fill_(0.75)

                # object alpha
                output_conv.weight[:, 4, :, :].fill_(0)
                output_conv.bias[4].fill_(0.0)

            # --- parts ---
            parts_start_idx = 2+n_sigma

            # part offsets (dx, dy)
            output_conv.weight[:, parts_start_idx: parts_start_idx + 2, :, :].fill_(0)
            output_conv.bias[parts_start_idx: parts_start_idx + 2].fill_(0)

            if n_sigma <= 2:
                # part sigmas (sx, sy)
                output_conv.weight[:, parts_start_idx + 2 : parts_start_idx + 2 + n_sigma, :, :].fill_(0)
                output_conv.bias[parts_start_idx + 2 : parts_start_idx + 2 + n_sigma].fill_(0.75)
            else:
                # part sigmas (sx, sy)
                output_conv.weight[:, parts_start_idx + 2 : parts_start_idx + 2 + 2, :, :].fill_(0)
                output_conv.bias[parts_start_idx + 2 : parts_start_idx + 2 + 2].fill_(0.75)

                # parts alpha
                output_conv.weight[:, parts_start_idx + 2 + 2 , :, :].fill_(0)
                output_conv.bias[parts_start_idx + 2 + 2].fill_(0.0)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)

        output_decoder = torch.cat([decoder.forward(output) for decoder in self.decoders], 1)

        return output_decoder
