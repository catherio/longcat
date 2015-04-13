require 'torch'
require 'nn'
require 'optim'

function train_model(model, criterion, data, labels, test_data, test_labels, opt)

    parameters, grad_parameters = model:getParameters()

    -- If GPU, then make space on the GPU device to copy data into
    if opt.type == 'cuda' then
       gpuBatch = torch.zeros(opt.minibatchSize+1, data:size(2), data:size(3))
       gpuLabels = torch.zeros(opt.minibatchSize+1)
       		 -- (the +1 is because minibatchSize is actually a bit of a fencepost error below...)
       
       gpuBatch = gpuBatch:cuda()
       gpuLabels = gpuLabels:cuda()
       
       test_data = test_data:cuda()
       test_labels = test_labels:cuda()
    end
    
    -- optimization functional to train the model with torch's optim library
    local function feval(x) 
        local minibatch = data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, data:size(2)):clone()
        local minibatch_labels = labels:sub(opt.idx, opt.idx + opt.minibatchSize):clone()

	if opt.type == 'cuda' then
	    gpuBatch:zero()
	    gpuLabels:zero()

	    gpuBatch[{}]  = minibatch
	    gpuLabels[{}] = minibatch_labels

	    model:training()
	    local minibatch_loss = criterion:forward(model:forward(gpuBatch), gpuLabels)
	    model:zeroGradParameters()
	    model:backward(gpuBatch, criterion:backward(model.output, gpuLabels))
	else
	    model:training()
            local minibatch_loss = criterion:forward(model:forward(minibatch), minibatch_labels)
            model:zeroGradParameters()
            model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
        end

        return minibatch_loss, grad_parameters
    end
    
    print("nEpochs=" .. opt.nEpochs .. ", nBatches=" .. opt.nBatches)
    for epoch=1,opt.nEpochs do
        local order = torch.randperm(opt.nBatches)
	startTime = sys.clock()
        for batch=1,opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
--            print("epoch: ", epoch, " batch: ", batch)
	    
	    if epoch == 1 and batch == 1 then
	       batchTime = sys.clock() - startTime
	       print("--> First batch time: " .. os.date("!%X", batchTime))
	       print("--> Estimated epoch time: " .. os.date("!%X", batchTime * opt.nBatches))
	       print("--> Estimated runtime: " .. os.date("!%X", batchTime * opt.nBatches * opt.nEpochs))
	    end
        end

        local accuracy = test_model(model, test_data, test_labels, opt)
	opt.trainlogger:add{['% mean class accuracy (train set)'] = accuracy}
        print("epoch ", epoch, " error: ", accuracy)

	local modelfile = paths.concat(opt.rundir, opt.modelName)
	torch.save(modelfile, model)
	
	epochTime = sys.clock() - startTime
	print("--> Epoch duration was " .. os.date("!%X", epochTime))
	print("--> " .. opt.nEpochs-epoch .. " remaining epochs, est. " .. os.date("!%X", epochTime * (opt.nEpochs-epoch)))

    end
end
