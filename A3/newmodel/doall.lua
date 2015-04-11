require 'torch'
require 'nn'
require 'optim'

-- Set up options
dofile 'opt.lua'

-- Put all the functions in the workspace; nothing happens yet
dofile 'load_glove.lua'
dofile 'preprocess_data.lua'
dofile 'train_model.lua'
dofile 'test_model.lua'

-- Load and preprocess
print("Loading word vectors...")
print(opt.glovePath)
print(opt.inputDim)
local glove_table = load_glove(opt.glovePath, opt.inputDim)

print("Loading raw data...")
local raw_data = torch.load(opt.dataPath)

print("Computing document input representations...")
-- local processed_data, labels = preprocess_data(raw_data, glove_table, opt)
local processed_data, labels = preprocess_better(raw_data, glove_table, opt)

-- Split data into makeshift training and validation sets
local training_data = processed_data:sub(1, opt.nClasses*opt.nTrainDocs, 1, processed_data:size(2)):clone()
local training_labels = labels:sub(1, opt.nClasses*opt.nTrainDocs):clone()

-- The test_data is the rest of the processed_data and labels
local test_data = processed_data:sub(opt.nClasses*opt.nTrainDocs+1,opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), 1, processed_data:size(2)):clone()
local test_labels = labels:sub(opt.nClasses*opt.nTrainDocs+1,opt.nClasses*(opt.nTrainDocs+opt.nTestDocs)):clone()

local model, criterion = opt.model()

print("Training model...")
train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
local results = test_model(model, test_data, test_labels)

print(results)
