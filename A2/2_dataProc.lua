----------------------------------------------------------------------
-- Data preprocessing
--
-- This script preprocesses the data.
-- 1, Load and reshape small or full unlabeled data for unsupervised learning
-- 2, Load and reshape train and test dataset
-- 3, Preprocess all datasets: zscore and normalize
--
-- Script structure borrowed from Clement Farabet
--
-- LongCat: Catherine Olsson, Long Sha, Kevin Brown
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
   cmd:option('-datafolder', 'dataset', 'subdirectory where dataset is saved')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- Note: files were converted from their original Matlab format
-- to Torch's internal format using the mattorch package. The
-- mattorch package allows 1-to-1 conversion between Torch and Matlab
-- files.

unlabel_file = opt.datafolder .. '/unlabel.dat'
train_file = opt.datafolder .. '/train.dat'
test_file = opt.datafolder .. '/test.dat'

----------------------------------------------------------------------
-- unlabeled/training/test size

if opt.size == 'full' then
   print '==> using full unlabeled data, recommend use only for final testing'
   unsize = 100000
elseif opt.size == 'small' then
   print '==> using reduced unlabeled data, for fast experiments'
   unsize = 10000
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


-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.


-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,unlabelData:size() do
   unlabelData.data[i] = image.rgb2yuv(unlabelData.data[i])
end
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'

-- Normalize unlabeled data
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   unlabelData.data[{ {},i,{},{} }]:add(-unlabelData.data[{ {},i,{},{} }]:mean())
   unlabelData.data[{ {},i,{},{} }]:div(unlabelData.data[{ {},i,{},{} }]:std())
end

mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module,
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,unlabelData:size() do
      unlabelData.data[{ i,{c},{},{} }] = normalization:forward(unlabelData.data[{ i,{c},{},{} }])
   end
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do

   unlabelMean = unlabelData.data[{ {},i }]:mean()
   unlabelStd = unlabelData.data[{ {},i }]:std()

   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('unlabeled data, '..channel..'-channel, mean: ' .. unlabelMean)
   print('unlabeled data, '..channel..'-channel, standard deviation: ' .. unlabelStd)

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end
