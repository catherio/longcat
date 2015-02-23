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
   cmd:option('-datatransfer', 'hpc', 'how to get the data: local on hpc, or scp remotely: hpc | scp')
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
-- All download methods pull from our .dat files on HPC
-- because we have already converted those from binary to torch
-- format, so we do not have to rely on the user having
-- a working mattorch.

if (not paths.dir(opt.datafolder .. 'test.dat') or
	   not paths.dir(opt.datafolder .. 'train.dat') or 
	   not paths.dir(opt.datafolder .. 'unlabel.dat')) then
   -- Download only if not already there
   if opt.datatransfer == 'scp' then
	  print '==> SCPing data remotely from HPC. Please first open an HPC tunnel!'
	  os.execute('scp -r mercer:"/scratch/ls3470/DeepLearning/A2/*.dat" ./' .. opt.datafolder)
	  print '==> Done copying data from HPC'
   elseif opt.datatransfer == 'hpc' then
	  print '==> Copying data locally within HPC'
	  os.execute('cp -rv /scratch/ls3470/DeepLearning/A2/*.dat ./' .. opt.datafolder)
	  print '==> Done copying data locally  within HPC'
   end

end

----------------------------------------------------------------------
