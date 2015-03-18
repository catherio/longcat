----------------------------------------------------------------------
-- Model definition
--
-- This script defines the model to use in training.
--   + linear
--   + 2-layer neural network (MLP)
--   + convolutional network (ConvNet)
-- Script structure borrowed from Clement Farabet
--
-- LongCat: Catherine Olsson, Long Sha, Kevin Brown
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('STL-10 Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 10-class problem

if trainSur==0 then
    noutputs = 10
    -- input dimensions
    nfeats = 3
    width = 96
    height = 96
else
    noutputs = opt.nclasses
    nfeats = 3
    width = 32
    height = 32
end
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,128,256,128}
filtsize = opt.filtsize
poolsize = opt.poolsize
stride = 1;
padding = 0;
normkernel = image.gaussian1D(opt.normkernelsize)

----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'convnet' then

      -- a typical modern convolution network (conv+relu+pool)
      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize, stride, stride, padding))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize, stride, stride, padding))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 3 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsize, filtsize, stride, stride, padding))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      local calcDim = function(x)
      -- this equation computes the new dim of images given filtsize and padsize
      -- this does not work for stride ~=1, or padding ~=0
          return math.floor((x-(filtsize-1))/poolsize)
      end
      -- stage 3 : standard 2-layer neural network
      newDimW = calcDim(calcDim(calcDim(width)))
      newDimH = calcDim(calcDim(calcDim(height)))
      model:add(nn.View(nstates[3]*newDimW*newDimH))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(nstates[3]*newDimW*newDimH, nstates[4]))
      model:add(nn.ReLU())
      model:add(nn.Linear(nstates[4], noutputs))

else

   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)
