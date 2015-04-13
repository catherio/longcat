-- Here is where to change parameters!! --

-- Configuration parameters
opt = {}

-- Name of the model.net output file
opt.modelName = 'model.net'

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
--opt.nEpochs = 5
opt.nEpochs = 50
opt.minibatchSize = 128
opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
opt.learningRate = 0.05
opt.learningRateDecay = 0.001
opt.momentum = 0.2
opt.idx = 1

-- Handmade SGD parameters
opt.learningRateCut = true
opt.learningRateCutAmount = 0.5
opt.learningRateCutPeriod = 3

-- Which preprocessing should we use? (both load a "preprocess_data()")
-- dofile('preprocess_glove_plain.lua')
dofile('preprocess_glove_plain_100.lua')

-- Which model should we use? (both load a "get_model()")
-- dofile('model_baseline.lua')
-- dofile('model_deep_zhanglike.lua')
-- dofile('model_deep_take2.lua')
dofile('model_shallower.lua')
opt.model, opt.criterion = get_model()

-- to CUDA or not to CUDA?
opt.type = 'cuda'
if opt.type == 'cuda' then
   require('cunn')
   torch.setdefaulttensortype('torch.FloatTensor')
   opt.model:cuda()
   opt.criterion:cuda()
end

