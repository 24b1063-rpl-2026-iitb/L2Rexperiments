import torch.nn as nn
from attr import asdict

from allrank.models.transformer import make_transformer
from allrank.models.feature_attention import IntraFeatureAttention
from allrank.utils.python_utils import instantiate_class


def first_arg_id(x, *y):
    return x


class FCModel(nn.Module):
    """
    This class represents a fully connected neural network model with given layer sizes and activation function.
    """
    def __init__(self, sizes, input_norm, activation, dropout, n_features):
        """
        :param sizes: list of layer sizes (excluding the input layer size which is given by n_features parameter)
        :param input_norm: flag indicating whether to perform layer normalization on the input
        :param activation: name of the PyTorch activation function, e.g. Sigmoid or Tanh
        :param dropout: dropout probability
        :param n_features: number of input features
        """
        super(FCModel, self).__init__()
        sizes.insert(0, n_features)
        layers = [nn.Linear(size_in, size_out) for size_in, size_out in zip(sizes[:-1], sizes[1:])]
        self.input_norm = nn.LayerNorm(n_features) if input_norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else instantiate_class(
            "torch.nn.modules.activation", activation)
        self.dropout = nn.Dropout(dropout or 0.0)
        self.output_size = sizes[-1]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the FCModel.
        :param x: input of shape [batch_size, slate_length, self.layers[0].in_features]
        :return: output of shape [batch_size, slate_length, self.output_size]
        """
        x = self.input_norm(x)
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        return x


class LTRModel(nn.Module):
    """
    This class represents a full neural Learning to Rank model with a given encoder model.
    """
    def __init__(self, input_layer, encoder, output_layer):
        """
        :param input_layer: the input block (e.g. FCModel)
        :param encoder: the encoding block (e.g. transformer.Encoder)
        :param output_layer: the output block (e.g. OutputLayer)
        """
        super(LTRModel, self).__init__()
        self.input_layer = input_layer if input_layer else nn.Identity()
        self.encoder = encoder if encoder else first_arg_id
        self.output_layer = output_layer

    def prepare_for_output(self, x, mask, indices):
        """
        Forward pass through the input layer and encoder.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: encoder output of shape [batch_size, slate_length, encoder_output_dim]
        """
        if isinstance(self.input_layer, IntraFeatureAttention):
            x_input = self.input_layer(x, mask)
        else:
            x_input = self.input_layer(x)
        return self.encoder(x_input, mask, indices)

    def forward(self, x, mask, indices):
        """
        Forward pass through the whole LTRModel.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: model output of shape [batch_size, slate_length, output_dim]
        """
        chunk_size = 16
        if not self.training and x.size(0) > chunk_size and x.size(1) > 256:
            import torch
            outputs = []
            for i in range(0, x.size(0), chunk_size):
                sub_x = x[i : i + chunk_size]
                sub_mask = mask[i : i + chunk_size]
                sub_indices = indices[i : i + chunk_size]
                outputs.append(self.output_layer(self.prepare_for_output(sub_x, sub_mask, sub_indices)))
            return torch.cat(outputs, dim=0)
        return self.output_layer(self.prepare_for_output(x, mask, indices))

    def score(self, x, mask, indices):
        """
        Forward pass through the whole LTRModel and item scoring.

        Used when evaluating listwise metrics in the training loop.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: scores of shape [batch_size, slate_length]
        """
        chunk_size = 16
        if x.size(0) > chunk_size and x.size(1) > 256:
            import torch
            scores = []
            for i in range(0, x.size(0), chunk_size):
                sub_x = x[i : i + chunk_size]
                sub_mask = mask[i : i + chunk_size]
                sub_indices = indices[i : i + chunk_size]
                scores.append(self.output_layer.score(self.prepare_for_output(sub_x, sub_mask, sub_indices)))
            return torch.cat(scores, dim=0)
        return self.output_layer.score(self.prepare_for_output(x, mask, indices))


class OutputLayer(nn.Module):
    """
    This class represents an output block reducing the output dimensionality to d_output.
    """
    def __init__(self, d_model, d_output, output_activation=None):
        """
        :param d_model: dimensionality of the output layer input
        :param d_output: dimensionality of the output layer output
        :param output_activation: name of the PyTorch activation function used before scoring, e.g. Sigmoid or Tanh
        """
        super(OutputLayer, self).__init__()
        self.activation = nn.Identity() if output_activation is None else instantiate_class(
            "torch.nn.modules.activation", output_activation)
        self.d_output = d_output
        self.w_1 = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Forward pass through the OutputLayer.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length, self.d_output]
        """
        return self.activation(self.w_1(x).squeeze(dim=2))

    def score(self, x):
        """
        Forward pass through the OutputLayer and item scoring by summing the individual outputs if d_output > 1.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length]
        """
        if self.d_output > 1:
            return self.forward(x).sum(-1)
        else:
            return self.forward(x)


def make_model(fc_model, transformer, post_model, n_features):
    """
    Helper function for instantiating LTRModel.
    :param fc_model: FCModel or IntraFeatureAttention used as input block
    :param transformer: transformer Encoder used as encoder block
    :param post_model: parameters dict for OutputModel output block (excluding d_model)
    :param n_features: number of input features
    :return: LTR model instance
    """
    if fc_model:
        if isinstance(fc_model, dict) and fc_model.get("intra_feature"):
            intra_cfg = fc_model.get("intra_feature")
            d_model = fc_model.get("sizes", [144])[-1]
            n_heads = intra_cfg.get("n_heads", 4) if isinstance(intra_cfg, dict) else 4
            d_ff = intra_cfg.get("d_ff", 288) if isinstance(intra_cfg, dict) else 288
            dropout = intra_cfg.get("dropout", 0.1) if isinstance(intra_cfg, dict) else 0.1
            chunk_size = intra_cfg.get("chunk_size", 1024) if isinstance(intra_cfg, dict) else 1024
            mask_aware = intra_cfg.get("mask_aware", True) if isinstance(intra_cfg, dict) else True
            fc_model = IntraFeatureAttention(
                n_features=n_features,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                chunk_size=chunk_size,
                mask_aware=mask_aware
            )
        else:
            fc_dict = {k: v for k, v in fc_model.items() if k != "intra_feature"} if isinstance(fc_model, dict) else fc_model
            fc_model = FCModel(**fc_dict, n_features=n_features)  # type: ignore
    d_model = n_features if not fc_model else fc_model.output_size
    if transformer:
        transformer = make_transformer(n_features=d_model, **asdict(transformer, recurse=False))  # type: ignore
    model = LTRModel(fc_model, transformer, OutputLayer(d_model, **post_model))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
