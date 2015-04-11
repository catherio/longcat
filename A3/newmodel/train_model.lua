require 'torch'
require 'nn'
require 'optim'

function train_model(model, criterion, data, labels, test_data, test_labels, opt)

    parameters, grad_parameters = model:getParameters()
    
    -- optimization functional to train the model with torch's optim library
    local function feval(x) 
        local minibatch = data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, data:size(2)):clone()
        local minibatch_labels = labels:sub(opt.idx, opt.idx + opt.minibatchSize):clone()
        
        model:training()
        local minibatch_loss = criterion:forward(model:forward(minibatch), minibatch_labels)
        model:zeroGradParameters()
        model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
        
        return minibatch_loss, grad_parameters
    end
    
    for epoch=1,opt.nEpochs do
        local order = torch.randperm(opt.nBatches) -- not really good randomization, TODO fix it?
        for batch=1,opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
            --print("epoch: ", epoch, " batch: ", batch)
        end

        local accuracy = test_model(model, test_data, test_labels, opt)
        print("epoch ", epoch, " error: ", accuracy)

    end
end
