
#import sys
#import pathlib

# Ensure the project's `src/` directory is on sys.path so package imports work
# when running this script directly (without installing the package).
#sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

modelName = 'speechBaseline4'

args = {}
args['outputDir'] = '../models/' + modelName
args['datasetPath'] = '../data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
args['optimizer'] = 'ADAM'
args['warmupSteps'] = 500
args['decayType'] = 'linear'
args['lrStart'] = 0.05
args['lrEnd'] = 0.02
args['optimizerEps'] = 0.1
args['SGDMomentum'] = 0
args['l2_decay'] = 1e-5
args['nUnits'] = 256
args['nBatch'] = 10000 #3000
args['patience'] = 10000 # Reduce to implement early stopping
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256 # Number of electrodes * 2
args['nThresholdCrossings'] = 128
args['nSpikeBandPowers'] = 128
args['dropout'] = 0.2
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False

# --- Time masking hyperparameters ---
args['timeMaskNum'] = 20        # N: number of masks per trial
args['timeMaskMaxFrac'] = 0.075 # M: max mask length as fraction of trial length

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)