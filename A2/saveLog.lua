----------------------------------------------------------------------
-- This script updates train log and test log, and saves model
--
-- Script structure borrowed from Clement Farabet
--
-- LongCat: Catherine Olsson, Long Sha, Kevin Brown
----------------------------------------------------------------------

-- update logger/plot
trainLogger:add{['% mean class accuracy (train set)'] = aveTrainAcc}
if opt.plot then
  trainLogger:style{['% mean class accuracy (train set)'] = '-'}
  trainLogger:plot()
end

-- save/log current net
local filename = paths.concat(opt.save, 'model.net')
os.execute('mkdir -p ' .. sys.dirname(filename))
print('==> saving model to '..filename)
torch.save(filename, model)

-- update log/plot
testLogger:add{['% mean class accuracy (test set)'] = aveValAcc}

if opt.plot then
  testLogger:style{['% mean class accuracy (test set)'] = '-'}
  testLogger:plot()
end
