-- Here is where to change parameters!! --

-- Configuration parameters
opt = {}

-- word vector dimensionality
-- must be 50, 100, 200, or 300
opt.inputDim = 50

-- change these to the appropriate data locations
opt.glovePath = "/scratch/courses/DSGA1008/A3/glove/glove.6B." .. opt.inputDim .. "d.txt" -- path to raw glove data .txt file
opt.dataPath = "/scratch/courses/DSGA1008/A3/data/train.t7b"

-- nTrainDocs is the number of documents per class used in the training set, i.e.
-- here we take the first nTrainDocs documents from each class as training samples
-- and use the rest as a validation set.
opt.nTrainDocs = 10000
opt.nTestDocs = 1000
opt.nClasses = 5

-- SGD parameters - play around with these
opt.nEpochs = 5
opt.minibatchSize = 128
opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
opt.learningRate = 0.1
opt.learningRateDecay = 0.001
opt.momentum = 0.1
opt.idx = 1

-- Which model should we use?
dofile('model_baseline.lua')
opt.model = model_baseline
