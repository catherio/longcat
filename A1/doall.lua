----------------------------------------------------------------------
-- This tutorial shows how to train different models on the street
-- view house number dataset (SVHN),
-- using multiple optimization techniques (SGD, ASGD, CG), and
-- multiple types of models.
--
-- This script demonstrates a classical example of training
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem.
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
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
cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
cmd:option('-gausswidth', 2, 'sigma_horz for gaussian window preprocessing')
-- model:
cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
cmd:option('-filtsize', 5, 'filter size, default = 5')
cmd:option('-poolsize', 2, 'pooling size, default = 2')
cmd:option('-normkernelsize', 7, 'gaussian kernel size for normalization, default = 7')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'double', 'type: double | float | cuda')
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

dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print '==> training!'

aveacc= 0
aveTable = {ave={}}
continue = true
while continue do
   train()
   test()
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

resultLogger = {id={},prediction={}}
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
   table.insert(resultLogger.id,target)
   table.insert(resultLogger.prediction,ptarget[1])
end

p={resultPath=opt.save..'/results.csv'}
csvigo.save{data=resultLogger,path=p.resultPath}
