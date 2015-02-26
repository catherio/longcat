----------------------------------------------------------------------
-- Doall for STL-10
--
-- Script structure borrowed from Clement Farabet
--
-- LongCat: Catherine Olsson, Long Sha, Kevin Brown
----------------------------------------------------------------------

require 'csvigo'

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
-- model:
cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
cmd:option('-filtsize', 5, 'filter size, default = 5')
cmd:option('-poolsize', 2, 'pooling size, default = 2')
cmd:option('-normkernelsize', 7, 'gaussian kernel size for normalization, default = 7')
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
cmd:option('-validation', 'cross', 'method for validation: a single validation set, cross validation, or use the test set directly: sub | cross | test')
cmd:option('-valratio', 1/4, 'validation set ratio compared to training, for subset validation only')
cmd:option('-totalSplit', 10, 'number of validation sets in training, for cross-validation only')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile '1_dataGet.lua' --will only re-download if necessary
dofile '2_dataProc.lua'
-- dofile 2.5_unsupervised.lua
-- dofile 2.75_applyUnsupervised.lua? Or is this
dofile '3_model.lua'
dofile '4_loss.lua'
dofile '5_train.lua' --creates a function called train()
dofile '6_test.lua' --creates a function called test()

----------------------------------------------------------------------
print '==> training!'

aveacc= 0
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

    table.insert(aveTable.ave,aveacc)
    n=#aveTable.ave
    --apply a random criterion:
    --at least 150 epochs, improvement < 0
    if n >= 150 then
        ave=torch.Tensor(aveTable.ave)
        c=ave[n]-ave[n-1]
        if c < 0 then continue=false end
    end
end
