import torch
from torch import nn

from neural_decoder.augmentations import GaussianSmoothing, SpeckleNoise


# Add TDS blocks class
class TDSConvBlock(nn.Module):
    """
    TDS conv block:
    time-wise conv -> depthwise temporal conv -> pointwise conv -> ReLU -> Dropout -> Residual -> LayerNorm
    """
    def __init__(self, channels, kernel_size=5, dropout=0.2):
        super().__init__()
        # Time-wise conv mixes channels over time
        self.tw = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1,
            bias=True
        )
        # Depthwise conv captures per-channel temporal motifs
        self.dw = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,      # depthwise along time
            bias=True
        )
        # Pointwise conv mixes across channels
        self.pw = nn.Conv1d(
            channels, channels,
            kernel_size=1,
            bias=True
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):  # x: (B, C, T)
        res = x
        x = self.tw(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(x)
        x = self.drop(x)
        x = x + res  # residual

        # LayerNorm expects (B, T, C), so transpose twice
        x = x.transpose(1, 2)   # (B, T, C)
        x = self.ln(x)
        x = x.transpose(1, 2)   # (B, C, T)
        return x


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,

        # Conditional logic for choosing whether to use TDS blocks or not
        use_tds=True,
        num_tds_blocks=3,
        tds_channels=None,

    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign() 
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        # Added for conditional logic
        self.use_tds = use_tds
        self.num_tds_blocks = num_tds_blocks

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)


        # Conditional usage of TDS blocks
        if not self.use_tds:
            self.unfolder = torch.nn.Unfold(
                (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
            )
            gru_input_dim = (neural_dim) * self.kernelLen

        else: # TDS Conv
            self.tds_channels = tds_channels or neural_dim

            # first temporal conv that also does the old stride/kernel downsampling
            self.tds_in = nn.Conv1d(
                in_channels=neural_dim,
                out_channels=self.tds_channels,
                kernel_size=self.kernelLen,
                stride=self.strideLen,
                padding=0,          # no padding keeps length formula unchanged
                bias=True
            )

            # stack a few TDS blocks
            self.tds_blocks = nn.Sequential(
                *[TDSConvBlock(self.tds_channels, kernel_size=5, dropout=self.dropout)
                for _ in range(self.num_tds_blocks)]
            )

            gru_input_dim = self.tds_channels

        # GRU layers
        self.gru_decoder = nn.GRU(
            gru_input_dim, 
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )
       
        # New structure: GRU → LayerNorm → FC1 → ReLU → Dropout → FC2 → logits
        # rnn outputs
        if self.bidirectional:
            gru_output_dim = hidden_dim * 2
        else:
            gru_output_dim = hidden_dim

        # --- Post-GRU normalization + MLP head ---
        self.post_ln = nn.LayerNorm(gru_output_dim)                 # LayerNorm on GRU outputs
        self.post_fc1 = nn.Linear(gru_output_dim, gru_output_dim)   # First FC layer, same dim
        self.post_dropout = nn.Dropout(self.dropout)                # Dropout for regularization
        self.post_fc2 = nn.Linear(gru_output_dim, n_classes + 1)    # Final FC layer for mapping to class logits (+1 for CTC blank)


    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # Conditional logic
        if self.use_tds:
            # TDS conv pipeline
            # transformedNeural: (B, T, F)
            x = transformedNeural.permute(0, 2, 1)  # (B, F, T)
            x = self.tds_in(x)        # (B, C, T_out)  C = tds_channels
            x = self.tds_blocks(x)    # (B, C, T_out)
            stridedInputs = x.permute(0, 2, 1)  # (B, T_out, C)

        else: 
            # Baseline Unfold
            stridedInputs = torch.permute(
                self.unfolder(
                    torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
                ),
                (0, 2, 1),
            )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())   # shape: (batch, T_out, gru_output_dim)
    
        # --- LayerNorm → FC1 → ReLU → Dropout → FC2 ---

        out = self.post_ln(hid)         # Normalize GRU outputs across features for each time step
        out = self.post_fc1(out)        # First FC layer
        out = torch.relu(out)
        out = self.post_dropout(out)
        seq_out = self.post_fc2(out)    # Final FC layer to produce class logits per time step

        return seq_out

class LSTMDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(LSTMDecoder, self).__init__()

        # similar to GRUDecoder
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )

        # Gaussian smoothing over time 
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        # Day-specific linear transform 
        self.dayWeights = torch.nn.Parameter(
            torch.randn(nDays, neural_dim, neural_dim)
        )
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # per-day input layers 
        for x in range(nDays):
            setattr(self, f"inpLayer{x}", nn.Linear(neural_dim, neural_dim))
        for x in range(nDays):
            thisLayer = getattr(self, f"inpLayer{x}")
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # LSTM layers
        self.lstm_decoder = nn.LSTM(
            neural_dim * self.kernelLen,  
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        
        for name, param in self.lstm_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        for name, param in self.lstm_decoder.named_parameters():
            if "bias_ih" in name:
                hidden_size = param.size(0) // 4
                #forget gate bias to 1.0
                param.data[hidden_size : 2 * hidden_size].fill_(1.0)

        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1  # +1 for CTC blank
            )
        else:
            self.fc_decoder_out = nn.Linear(
                hidden_dim, n_classes + 1  # +1 for CTC blank
            )

    def forward(self, neuralInput, dayIdx):
        """
        neuralInput: [B, T, C]  (batch, time, features)
        dayIdx:      [B]        (which recording day for each trial)
        """
        
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)   # [B, C, C]
        dayBias = torch.index_select(self.dayBias, 0, dayIdx)         # [B, 1, C]

        transformedNeural = (
            torch.einsum("btd,bdk->btk", neuralInput, dayWeights) + dayBias
        )
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )  # [B, T', C * kernelLen]

        # Initial hidden and cell states
        num_directions = 2 if self.bidirectional else 1
        batch_size = transformedNeural.size(0)

        h0 = torch.zeros(
            self.layer_dim * num_directions,
            batch_size,
            self.hidden_dim,
            device=self.device,
        ).requires_grad_()

        c0 = torch.zeros(
            self.layer_dim * num_directions,
            batch_size,
            self.hidden_dim,
            device=self.device,
        ).requires_grad_()

        # LSTM forward
        hid, _ = self.lstm_decoder(stridedInputs, (h0.detach(), c0.detach()))

        
        seq_out = self.fc_decoder_out(hid)  
        return seq_out
