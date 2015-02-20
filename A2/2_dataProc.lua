----------------------------------------------------------------------
----------------------------------------------------------------------
-- Data preprocessing
--
-- This script preprocesses the data.
-- 1, Load and reshape small/full unlabeled data for unsupervised learning
-- 2, Load and reshape train/test dataset
-- 3, todo: Create validation set from training data
-- 4, todo: Preprocess all datasets: zscore
--
-- Script structure borrowed from Clement Farabet
--
-- LongCat: Catherine Olsson, Long Sha, Kevin Brown
----------------------------------------------------------------------
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('STL-10 Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'small', 'how many unlabeled samples do we load: small | full')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- Note: files were converted from their original Matlab format
-- to Torch's internal format using the mattorch package. The
-- mattorch package allows 1-to-1 conversion between Torch and Matlab
-- files.

unlabel_file = 'unlabel.dat'
train_file = 'train.dat'
test_file = 'test.dat'

----------------------------------------------------------------------
-- unlabeled/training/test size

if opt.size == 'full' then
   print '==> using full unlabeled data, recommend use only for final testing'
   unsize = 100000
elseif opt.size == 'small' then
   print '==> using reduced unlabeled data, for fast experiments'
   unsize = 20000
end
trsize = 5000
tesize = 8000

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, and re-arrange it to be compatible
-- with Torch's representation. Matlab uses a column-major representation,
-- Torch is row-major, so we just have to transpose the data.

-- Note: the data, in X, is 2-d: the 1st dim indexes images in columns, the 2nd
-- dim indexes the samples. After transpose the data, we reshape the data into
-- 4-d: 1st dimension indexes samples, 2nd dim is color channel (RGB), 3rd and
-- 4th dim is 96*96 images. We further transpose images to create images in
-- normal view

-- load the training data.

-- load the unlabel data.
loaded = torch.load(unlabel_file,'ascii')
loaded.X = loaded.X[{{},{1,unsize}}]

unlabelData = {
   data = loaded.X:transpose(1,2):reshape(unsize,3,96,96):transpose(3,4),
   size = function() return unsize end
}


loaded = torch.load(train_file,'ascii')
trainData = {
   data = loaded.X:transpose(1,2):reshape(trsize,3,96,96):transpose(3,4),
   labels = loaded.y[1],
   size = function() return trsize end
}

-- load the test data.

loaded = torch.load(test_file,'ascii')
testData = {
   data = loaded.X:transpose(1,2):reshape(tesize,3,96,96):transpose(3,4),
   labels = loaded.y[1],
   size = function() return tesize end
}


----------------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch,
-- in general by doing: dst = src:type('torch.TypeTensor'),
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

unlabelData.data = unlabelData.data:float()
trainData.data = trainData.data:float()
testData.data = testData.data:float()


-- todo: data preprocessing!!
