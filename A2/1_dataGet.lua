----------------------------------------------------------------------
-- Data loading
--
-- This script acquires the data.
--
-- This script can be run with the interactive mode:
-- $ torch -i 1_data.lua
--
-- Script structure borrowed from Clement Farabet
--
-- LongCat: Catherine Olsson, Long Sha, Kevin Brown
----------------------------------------------------------------------

require 'torch'   -- torch

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('STL-10 Dataset Acquisition')
   cmd:text()
   cmd:option('-dataloc', 'hpc', 'location from which to get the data: hpc | www')
   cmd:option('-datafolder', 'dataset', 'subdirectory to save dataset in')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- Create data directory
if not paths.dir(opt.datafolder) then
   paths.mkdir(opt.datafolder)
end

----------------------------------------------------------------------
-- Download data
if opt.dataloc == 'hpc' then
   print '==> Copying data off HPC'
   local filename = paths.concat(opt.datafolder, 'temp.t7')
   torch.save(filename, torch.FloatTensor({1, 2, 3}))

elseif opt.dataloc == 'www' then
   print '==> Downloading data from www'
   local filename = paths.concat(opt.datafolder, 'temp.t7')
   torch.save(filename, torch.FloatTensor({1, 2, 3}))
end

----------------------------------------------------------------------
