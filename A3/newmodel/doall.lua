require 'torch'
require 'nn'
require 'optim'

-- Set up options
dofile 'opt.lua'

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

model, criterion = get_model()

print("Training model...")
print(training_data:size())
train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
results = test_model(model, test_data, test_labels)

print(results)
