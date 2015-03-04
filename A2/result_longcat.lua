require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'cunn'
require 'csvigo'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 1, 'number of threads')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-datafolder', 'dataset', 'subdirectory where dataset is saved')
cmd:option('-datatransfer', 'hpc', 'how to get the data: local on hpc, or scp remotely: hpc | scp')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.998, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'double', 'type: double | float | cuda')
-- validation
cmd:option('-validation', 'sub', 'method for validation: a single validation set, cross validation, or use the test set directly: sub | cross | test')
cmd:option('-valratio', 1/4, 'validation set ratio compared to training, for subset validation only')
cmd:option('-totalSplit', 10, 'number of validation sets in training, for cross-validation only')

cmd:text()
opt = cmd:parse(arg or {})


--Here we need to apply the correct method, for running model through image patches
dofile '2_dataProc.lua'
dofile '8_featuremap.lua'


-- create supervised learning mode
print '==>initializing for supervised learning'
trainSur = 0
dofile '4_loss.lua'
dofile '5_train.lua' --creates a function called train()
dofile '6_test.lua' --creates a function called test()

----------------------------------------------------------------------
print '==> training!'

aveValAcc= 0
aveTable = {ave={}}
continue = true
while continue do
-- add validation method
    if opt.validation=='cross' then
        aveValTemp=torch.zeros(opt.totalSplit,1)
        aveTrainTemp=torch.zeros(opt.totalSplit,1)
        for curSplit = 0,opt.totalSplit-1 do
            print ('==> training Set: ' ..curSplit)
            train()
            print ('==> validating Set: ' ..curSplit)
            test()
            aveTrainTemp[curSplit+1]=aveTrainAcc
            aveValTemp[curSplit+1]=aveValAcc
        end
        aveTrainAcc = aveTrainTemp:mean()
        aveValAcc = aveValTemp:mean()
    else
        train()
        test()
    end
    dofile 'saveLog.lua'

    table.insert(aveTable.ave,aveValAcc)
    n=#aveTable.ave
    --apply a random criterion:
    --at least 100 epochs, improvement < 0
    if n >= 500 then
        ave=torch.Tensor(aveTable.ave)
        c=ave[n]-ave[n-1]
        if c < 0 then continue=false end
    end
end


--start resultLogger for the final output
resultLogger = {prediction={},id={}}
model:evaluate()
for t = 1,testData:size() do
   input = testData.data[t]
   target = testData.labels[t]
   if opt.type == 'double' then input = input:double()
   elseif opt.type == 'cuda' then input = input:cuda() end
   -- test sample
   pred = model:forward(input)
   --save predicted label based on max(nll), will change based on criterion
   m,ptarget = pred:max(1)
   table.insert(resultLogger.prediction,ptarget[1])
   table.insert(resultLogger.id,t)
end

--save
p={resultPath=paths.concat(opt.save,'./predictions.csv')}
csvigo.save{data=resultLogger,path=p.resultPath}
