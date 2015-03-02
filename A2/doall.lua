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
cmd:option('-threads', 1, 'number of threads')
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
cmd:option('-weightDecay', 0.5, 'weight decay (SGD only)')
cmd:option('-momentum', 0.998, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'double', 'type: double | float | cuda')
-- validation
cmd:option('-validation', 'cross', 'method for validation: a single validation set, cross validation, or use the test set directly: sub | cross | test')
cmd:option('-valratio', 1/4, 'validation set ratio compared to training, for subset validation only')
cmd:option('-totalSplit', 10, 'number of validation sets in training, for cross-validation only')
-- surrogate data
cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
cmd:option('-nclasses', 20, 'how many surrogate classes do we create, default = 5')
cmd:option('-nexemplars', 100, 'how many exemplars in each surrogate class, default = 10')
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
dofile '7_surrogateGen.lua' --generate surrograte dataset
trainSur = 1
dofile '3_model.lua'
dofile '4_loss.lua'
dofile '5_train.lua' --creates a function called train()
dofile '6_test.lua' --creates a function called test()
print '==> training surrogate dataset!'

aveValacc= 0
aveTable = {ave={}}
continue = true
while continue do
    train()
    test()
    dofile 'saveLog.lua'
    table.insert(aveTable.ave,aveValacc)
    n=#aveTable.ave
    --apply a random criterion:
    --at least 100 epochs, improvement < 0
    if n >= 100 then
        ave=torch.Tensor(aveTable.ave)
        c=ave[n]-ave[n-1]
        if c < 0 then continue=false end
    end
end


filename = paths.concat(opt.save, modelname)
surmodel = torch.load(filename)
convertModel = nn.Sequential()
convertModel:add(surmodel:get(1))
convertModel:add(surmodel:get(2))
convertModel:add(surmodel:get(3))
convertModel:add(surmodel:get(4))
convertModel:add(surmodel:get(5))
convertModel:add(surmodel:get(6))

convertModel:evaluate()

t = trainData.data[1]
if opt.type == 'double' then t = t:double()
elseif opt.type == 'cuda' then t = t:cuda() end
p = convertModel:forward(t)
-- create new training and test dataset by forwarding data into converModel
newTrain = {data=torch.Tensor(trsize,p:size(1),p:size(2),p:size(3)),
            labels=trainData.labels,
            size = function() return trsize end
           }
newTest = {data=torch.Tensor(tesize,p:size(1),p:size(2),p:size(3)),
            labels=testData.labels,
            size = function() return tesize end
          }

print '==> forwarding model on surrogate data'
for i=1,trsize do
    local sample = trainData.data[i]
    if opt.type == 'double' then sample = sample:double()
    elseif opt.type == 'cuda' then sample = sample:cuda() end
    outsample=convertModel:forward(sample)
    --if ran cuda on model, convert cuda back to double
    newTrain.data[i] = outsample:double()
end
for i=1,tesize do
    local sample = testData.data[i]
    if opt.type == 'double' then sample = sample:double()
    elseif opt.type == 'cuda' then sample = sample:cuda() end
    outsample=convertModel:forward(sample)
    --if ran cuda on model, convert cuda back to double
    newTest.data[i] = outsample:double()
end
-- overwrite current trainData and testData
print '==> overwrite current trainData and testData'
trainData=newTrain
testData=newTest
filename = paths.concat(opt.save, 'newTrain.dat')
torch.save(filename, trainData,'ascii')
filename = paths.concat(opt.save, 'newTest.dat')
torch.save(filename, testData,'ascii')

-- construct a simple model for supervised learning
print '==>constructing simple linear model for supervised learning'
model = nn.Sequential()
model:add(nn.Reshape(p:size(1)*p:size(2)*p:size(3)))
model:add(nn.Linear(p:size(1)*p:size(2)*p:size(3),10))

-- create supervised learning mode
print '==>initializing for supervised learning'
trainSur = 0
dofile '4_loss.lua'
dofile '5_train.lua' --creates a function called train()
dofile '6_test.lua' --creates a function called test()

----------------------------------------------------------------------
print '==> training!'

aveValacc= 0
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

    table.insert(aveTable.ave,aveValacc)
    n=#aveTable.ave
    --apply a random criterion:
    --at least 100 epochs, improvement < 0
    if n >= 100 then
        ave=torch.Tensor(aveTable.ave)
        c=ave[n]-ave[n-1]
        if c < 0 then continue=false end
    end
end
