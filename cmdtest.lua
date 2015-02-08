----------------------------------------------------------------------
-- This script simply outputs the value of a command line option,
-- as a way to test a shell script designed to parametrically
-- vary command line options.
--
-- Catherine Olsson
----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:option('-param', 1, 'arbitrary parameter')
cmd:text()
opt = cmd:parse(arg or {})

print(opt.param)
