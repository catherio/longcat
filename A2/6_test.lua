----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
-- require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on validation/test set:')

   -- run the testing loop
   for t = 1,valSet:size() do
      -- disp progress
      -- xlua.progress(t, valSet:size())

      -- get new sample
      local input = valSet.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = valSet.labels[t]

      -- test sample
      local pred = model:forward(input)
      -- print("\n" .. target .. "\n")
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   print("\n==> overall time = " .. (os.date("!%X",time)) .. 's')
   time = time / trainSet:size()
   print("==> number of training samples = " .. trainSet.size())
   print("==> time per sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   aveValAcc=confusion.totalValid*100

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   -- next iteration:
   confusion:zero()
end
