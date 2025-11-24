import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder, LSTMDecoder
from .dataset import SpeechDataset
from .augmentations import SpeckleNoise

def apply_time_mask(X: torch.Tensor, max_mask_len: int, n_masks: int) -> torch.Tensor:
    """
    SpecAugment-style time masking for neural time series.

    Args:
        X: [B, T, C] tensor (batch, time, channels)
        max_mask_len: maximum length (in time steps) of each mask
        n_masks: number of time masks per sequence

    Returns:
        X with some contiguous time segments zeroed out.
    """
    if max_mask_len <= 0 or n_masks <= 0:
        return X

    if X.dim() != 3:
        # We only expect [B, T, C] here; if not, just skip masking.
        return X

    B, T, C = X.shape
    if T == 0:
        return X

    max_mask_len = min(max_mask_len, T)

    # Work in-place on X; gradients still flow through the remaining (unmasked) entries.
    for b in range(B):
        for _ in range(n_masks):
            # Random mask length in [1, max_mask_len]
            L = int(torch.randint(1, max_mask_len + 1, (1,), device=X.device).item())
            if L >= T:
                t0 = 0
            else:
                # Random start index so that t0 + L <= T
                t0 = int(torch.randint(0, T - L + 1, (1,), device=X.device).item())

            X[b, t0 : t0 + L, :] = 0.0

    return X

def apply_time_mask_batch(X, X_len, n_masks, max_mask_frac):
    """
    Time-masking augmentation for GRU baseline.

    Args:
        X:        (B, T, C) tensor of neural features; B: batch size, T: max sequence length (padded), C: feature dim
        X_len:    (B,) tensor of true sequence lengths (no padding, in time bins).
        n_masks:  int, number of masks per trial (N in the paper).
        max_mask_frac: float, max mask length as fraction of trial length (M in the paper).

    Returns:
        X with contiguous time segments zeroed out within [0, X_len[b]) for each b.
    """
    if n_masks <= 0 or max_mask_frac <= 0:
        return X

    B, T, C = X.shape
    device = X.device

    for b in range(B):
        L = int(X_len[b].item())
        if L <= 0:
            continue

        F = int(max_mask_frac * L)  # max mask length in time bins
        if F <= 0:
            continue

        # For each mask, sample start S ~ U(0, L-F) and duration D ~ U(0, F)
        for _ in range(n_masks):
            max_start = max(L - F, 0)
            # start index
            if max_start > 0:
                S = torch.randint(0, max_start + 1, (1,), device=device).item()
            else:
                S = 0
            # duration (can be 0..F)
            D = torch.randint(0, F + 1, (1,), device=device).item()

            end = min(S + D, L)
            X[b, S:end, :] = 0.0

    return X

def getDatasetLoaders(
    datasetName,
    batchSize,
    num_threshold_crossings=128,
    num_spike_band_powers=128
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(
        loadedData["train"],
        transform=None,
        num_threshold_crossings=num_threshold_crossings,
        num_spike_band_powers=num_spike_band_powers
    )
    test_ds = SpeechDataset(
        loadedData["test"],
        num_threshold_crossings=num_threshold_crossings,
        num_spike_band_powers=num_spike_band_powers)

    # Only pin memory when CUDA is available to avoid warnings on CPU-only systems
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda" 

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
        num_threshold_crossings=args.get("nThresholdCrossings", 128),
        num_spike_band_powers=args.get("nSpikeBandPowers", 128)
    )

       # Choose RNN type (default to GRU if not specified)
    if "rnn_type" in args:
        rnn_type = str(args["rnn_type"]).lower()
    else:
        rnn_type = "gru"

    if rnn_type == "gru":
        ModelClass = GRUDecoder
    elif rnn_type == "lstm":
        ModelClass = LSTMDecoder
    else:
        raise ValueError(f"Unknown rnn_type: {rnn_type}. Use 'gru' or 'lstm'.")

    model = ModelClass(
        neural_dim=args["nInputFeatures"],
    model = GRUDecoder(
        neural_dim=args["nInputFeatures"] if (
            (args.get("nThresholdCrossings") is None) |
            (args.get("nSpikeBandPowers") is None)
        ) else args["nThresholdCrossings"] + args["nSpikeBandPowers"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        use_tds=False # Added for TDS conditional
    ).to(device)


    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    speckle_prob = float(args.get("speckle_prob", 0.0))
    speckle_noise = SpeckleNoise(p=speckle_prob).to(device)
    if args["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args["lrStart"],
            momentum=args.get("SGDMomentum", 0),
            weight_decay=args["l2_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=args['optimizerEps'],
            weight_decay=args["l2_decay"],
        )
    if args.get("warmupSteps", 0) > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1, total_iters=args["warmupSteps"])
        if args["decayType"] == 'cosine':
            decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args["nBatch"] - args["warmupSteps"]
            )
        else:
            decay_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=args["lrEnd"] / args["lrStart"],
                total_iters=args["nBatch"] - args["warmupSteps"],
            )
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay_scheduler], milestones=[args["warmupSteps"]])
    else:
        if args["decayType"] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args["nBatch"]
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=args["lrEnd"] / args["lrStart"],
                total_iters=args["nBatch"],
            )
  

    # --train--
    testLoss = []
    testCER = []

    patience = args.get("patience", np.inf)
    patience_counter = 0

    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )
        # Speckled masking / coordinated dropout on neural inputs 
        if speckle_prob > 0.0:
            # speckle_noise is an nn.Module living on the same device
            X = speckle_noise(X)

         #Time masking (SpecAugment-style) on neural inputs (training only)
        timeMaskMaxFrac = int(args.get("timeMask_maxLen", 0))
        timeMaskNum = int(args.get("timeMask_nMasks", 0))
        if timeMaskMaxFrac > 0 and timeMaskNum > 0:
            X = apply_time_mask(X, timeMaskMaxFrac, timeMaskNum)

        # Apply time masking augmentation
        n_masks = args.get("timeMaskNum", 0)
        max_mask_frac = args.get("timeMaskMaxFrac", 0.0)
        if n_masks > 0 and max_mask_frac > 0.0:
            X = apply_time_mask_batch(X, X_len, n_masks, max_mask_frac)

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        max_grad_norm = args["max_grad_norm"] if "max_grad_norm" in args else 0.0
        if max_grad_norm and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
                patience_counter = 0
            else:
                patience_counter += 1
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)
            
            if patience_counter == patience:
                break


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    # rnn_type might not exist for older runs → default to GRU
    if "rnn_type" in args:
        rnn_type = str(args["rnn_type"]).lower()
    else:
        rnn_type = "gru"

    if rnn_type == "gru":
        ModelClass = GRUDecoder
    elif rnn_type == "lstm":
        ModelClass = LSTMDecoder
    else:
        raise ValueError(f"Unknown rnn_type: {rnn_type}. Use 'gru' or 'lstm'.")

    model = ModelClass(
        neural_dim=args["nInputFeatures"],
    model = GRUDecoder(
        neural_dim=args["nInputFeatures"] if (
            (args.get("nThresholdCrossings") is None) |
            (args.get("nSpikeBandPowers") is None)
        ) else args["nThresholdCrossings"] + args["nSpikeBandPowers"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()