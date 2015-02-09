

require 'torch'
require 'csvigo'
require 'nn'      -- provides a normalization operator
require 'optim'   -- an optimization package, for online and batch methods
--require 'cunn'
opt={type='double'}


model = torch.load('./model.net')
test_file = 'test_32x32.t7'

tesize = 26032
loaded = torch.load(test_file,'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return tesize end
}

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

p={resultPath='./predictions.csv'}
csvigo.save{data=resultLogger,path=p.resultPath}
