require 'torch'
require 'nn'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
cmd:option('-momentum', 0.998, 'momentum (SGD only)')
cmd:option('-learningRateCutPeriod', 3, 'how often to halve learning rate')
newopt = cmd:parse(arg or {})

-- Set up options
dofile 'opt.lua'
opt.learningRate = newopt.learningRate
opt.momentum = newopt.momentum
opt.learningRateCutPeriod = newopt.learningRateCutPeriod

-- Put all the functions in the workspace; nothing happens yet
dofile 'load_glove.lua'

dofile 'train_model.lua'
dofile 'test_model.lua'

-- Load and preprocess
print("Loading word vectors...")
glove_table = load_glove(opt.glovePath, opt.inputDim)

print("Loading raw data...")
raw_data = torch.load(opt.dataPath)

print("Computing document input representations...")
processed_data, labels = preprocess_data(raw_data, glove_table, opt)

-- Split data into makeshift training and validation sets
training_data = processed_data:sub(1, opt.nClasses*opt.nTrainDocs, 1, processed_data:size(2)):clone()
training_labels = labels:sub(1, opt.nClasses*opt.nTrainDocs):clone()

-- The test_data is the rest of the processed_data and labels
test_data = processed_data:sub(opt.nClasses*opt.nTrainDocs+1,opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), 1, processed_data:size(2)):clone()
test_labels = labels:sub(opt.nClasses*opt.nTrainDocs+1,opt.nClasses*(opt.nTrainDocs+opt.nTestDocs)):clone()

-- If we got this far, then make a new directory for everything; copy opt into there
datestring = os.date('%m_%d_%X'):gsub(':(.+):', 'h%1m')
opt.rundir = './rundir_' .. datestring .. '/'

os.execute('mkdir ' .. opt.rundir)
os.execute('cp opt.lua ' .. opt.rundir)

opt.trainlogger = optim.Logger(opt.rundir .. datestring .. '_train.log')
opt.testlogger = optim.Logger(opt.rundir .. datestring .. '_test.log')

print('learningRate: ' .. opt.learningRate)
print('momentum: ' .. opt.momentum)
print('learningRateCutPeriod: ' .. opt.learningRateCutPeriod)

-- Do the thing!
print("Training model...")
print(training_data:size())
train_model(opt.model, opt.criterion, training_data, training_labels, test_data, test_labels, opt)

if opt.type == 'cuda' then
   test_data = test_data:cuda()
   test_labels = test_labels:cuda()
end

results = test_model(opt.model, test_data, test_labels)

opt.testlogger:add{['% mean class accuracy (test set)'] = results}

print(results)
