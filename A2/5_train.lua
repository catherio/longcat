----------------------------------------------------------------------
-- Training procedure
--
-- This script defines the training procedure, with options to...
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
--
-- Script structure borrowed from Clement Farabet
--
-- LongCat: Catherine Olsson, Long Sha, Kevin Brown
----------------------------------------------------------------------

require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:option('-validation', 'sub', 'method for validation, 1 subset, cross validation, or test set: sub | cross | test')
   cmd:option('-valratio', 1/4, 'validation set ratio compared to training')
   cmd:option('-totalSplit', 10, 'number of validation sets in training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- Validation procedure: Are we using test data, subset of training data, or cross validation?
if trainSur==0 then
    if opt.validation == 'test' then
        print '==> using test dataset for validation'
        valSet=testData
        trainSet=trainData
    elseif opt.validation == 'sub' then
    -- split trainData into valSet and the remaining trainSet
        print '==> using subset of training for validation'
        valsize = trsize * opt.valratio

        valSet={
          data = trainData.data[{{1,valsize},{},{},{}}],
          labels = trainData.labels[{{1,valsize}}],
          size = function() return valsize end
        }

        trainSet={
          data = trainData.data[{{valsize+1,-1},{},{},{}}],
          labels = trainData.labels[{{valsize+1,-1}}],
          size = function() return trsize-valsize end
        }

    elseif opt.validation == 'cross' then
    -- cross validation procedure is called outside this script, here we only split
    -- dataset once, depending on valratio and current split number
        print '==> using cross validation'
        if not curSplit then
            curSplit = 0
        end

        chunk = torch.floor(trsize * 1/opt.totalSplit)

    	-- fill validation set: take the nth chunk (simple slice)
        valSet={
          data = trainData.data[{{curSplit*chunk+1,curSplit*chunk+chunk},{},{},{}}],
          labels = trainData.labels[{{curSplit*chunk+1,curSplit*chunk+chunk}}],
          size = function() return chunk end
        }

    	-- fill training set. bipartite slicing is harder... just fill with
    	-- zeroes first, then populate
        trainSet={
          data = torch.zeros(trsize-chunk,3,96,96),
          labels = torch.zeros(trsize-chunk),
          size = function() return trsize-chunk end
        }

    	-- populate the training set in two slices
        trainSet.data[{{1,curSplit*chunk+1},{},{},{}}]=trainData.data[{{1,curSplit*chunk+1},{},{},{}}]
        trainSet.data[{{curSplit*chunk+1,-1},{},{},{}}]=trainData.data[{{curSplit*chunk+chunk+1,-1},{},{},{}}]
        trainSet.labels[{{1,curSplit*chunk+1}}]=trainData.labels[{{1,curSplit*chunk+1}}]
        trainSet.labels[{{curSplit*chunk+1,-1}}]=trainData.labels[{{curSplit*chunk+chunk+1,-1}}]
    end
else -- if trainSur==1
    print '==> using subset of surrogate dataset for validation'
    valsize = surData.data:size(1) * opt.valratio

    valSet={
      data = surData.data[{{1,valsize},{},{},{}}],
      labels = surData.labels[{{1,valsize}}],
      size = function() return valsize end
    }

    trainSet={
      data = surData.data[{{valsize+1,-1},{},{},{}}],
      labels = surData.labels[{{valsize+1,-1}}],
      size = function() return surData.data:size(1)-valsize end
    }

end

-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
if trainSur==0 then
    classes = {'1','2','3','4','5','6','7','8','9','0'}
else
    classes = {}
    for i=1,opt.nclasses do
        table.insert(classes,tostring(i))
    end
end
-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
if trainSur==0 then
    trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
    testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
    modelname = 'model.net'
else
    trainLogger = optim.Logger(paths.concat(opt.save, 'train_sur.log'))
    testLogger = optim.Logger(paths.concat(opt.save, 'test_sur.log'))
    modelname = 'model_sur.net'
end
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trainSet:size() * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trainSet:size())

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   -- parameters controlling when/how time estimates are displayed
   estimateStart = 5; -- make initial estimate at this training sample
   estimateInterval = 500; -- give updates after every nth training sample

   -- run the training loop
   for t = 1,trainSet:size(),opt.batchSize do

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainSet:size()) do
         -- load new sample
         local input = trainSet.data[shuffle[i]]
         local target = trainSet.labels[shuffle[i]]
         if opt.type == 'double' then input = input:double()
         elseif opt.type == 'cuda' then input = input:cuda() end
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end

	  -- estimate the total training time
	  if (t == estimateStart) or (t % estimateInterval == 0) then
	  	 timeSoFar = sys.clock() - time
		 print("==> interim time for "..t.." samples is "..
				  (os.date("!%X",timeSoFar)))
		 estimate = timeSoFar * (trainSet.size()/t)
		 print("    estimated time for "..trainSet.size().." samples is "..
				  (os.date("!%X",estimate)))
      end
   end

   -- time taken
   time = sys.clock() - time
   print("\n==> overall time = " .. (os.date("!%X",time)) .. 's')
   time = time / trainSet:size()
   print("==> number of training samples = " .. trainSet.size())
   print("==> time per sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   aveTrainAcc=confusion.totalValid*100


   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
