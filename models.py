import torch


class VeryTinyNeRFModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(VeryTinyNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 65 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims + self.viewdir_encoding_dims, filter_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MultiHeadNeRFModel(torch.nn.Module):
    r"""Define a "multi-head" NeRF model (radiance and RGB colors are predicted by
    separate heads).
    """

    def __init__(self, hidden_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(MultiHeadNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims, hidden_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 3_1 (default: 128 -> 1): Predicts radiance ("sigma")
        self.layer3_1 = torch.nn.Linear(hidden_size, 1)
        # Layer 3_2 (default: 128 -> 1): Predicts a feature vector (used for color)
        self.layer3_2 = torch.nn.Linear(hidden_size, hidden_size)

        # Layer 4 (default: 39 + 128 -> 128)
        self.layer4 = torch.nn.Linear(self.viewdir_encoding_dims + hidden_size, hidden_size)
        # Layer 5 (default: 128 -> 128)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 6 (default: 128 -> 3): Predicts RGB color
        self.layer6 = torch.nn.Linear(hidden_size, 3)

        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x, view = x[..., :self.xyz_encoding_dims], x[..., self.xyz_encoding_dims:]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return torch.cat((x, sigma), dim=-1)


class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(self, hidden_size=128):
        super(ReplicateNeRFModel, self).__init__()
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        self.layer1 = torch.nn.Linear(3 + 36, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer4 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer5 = torch.nn.Linear(hidden_size + 3 + 36, hidden_size)
        self.layer6 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer7 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer8 = torch.nn.Linear(hidden_size, 1 + hidden_size)
        self.layer9 = torch.nn.Linear(hidden_size + 3 + 36, hidden_size // 2)
        self.layer10 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.layer11 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.layer12 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.layer13 = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu
    
    def forward(self, x):
        xyz, direction = x[..., :3 + 36], x[..., 3 + 36:]
        xyz_encoded = xyz
        x = self.relu(self.layer1(xyz))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(torch.cat((xyz_encoded, x), dim=-1)))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.layer8(x)
        sigma = x[..., 0]
        feat = x[..., 1:]
        x = self.relu(self.layer9(torch.cat((feat, direction), dim=-1)))
        x = self.relu(self.layer10(x))
        x = self.relu(self.layer11(x))
        x = self.relu(self.layer12(x))
        rgb = self.layer13(x)
        return torch.cat((rgb, sigma.unsqueeze(-1)), dim=-1)


class FlexibleNeRFModel(torch.nn.Module):

    def __init__(self, num_layers=4, hidden_size=128, skip_connect_every=4,
                 num_encoding_functions=6, use_viewdirs=True):
        super(FlexibleNeRFModel, self).__init__()
        self.input_size = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs:
            self.input_size = 2 * self.input_size
        self.input_linear_layer = torch.nn.Linear(self.input_size, hidden_size)
        self.skip_connect_every = skip_connect_every
        self.linear_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.linear_layers.append(
                    torch.nn.Linear(self.input_size + hidden_size, hidden_size)
                )
            else:
                self.linear_layers.append(
                    torch.nn.Linear(hidden_size, hidden_size)
                )
        self.output_linear_layer = torch.nn.Linear(hidden_size, 4)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x_in = x.clone()
        x = self.input_linear_layer(x)
        for i in range(len(self.linear_layers)):
            if i % self.skip_connect_every == 0 and i > 0 and i != len(self.linear_layers) - 1:
                x = self.relu(torch.cat((x, x_in), dim=-1))
            x = self.relu(self.linear_layers[i](x))
        return self.output_linear_layer(x)
