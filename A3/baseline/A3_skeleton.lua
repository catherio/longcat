-------------------------------------------------------------------------
-- In this part of the assignment you will become more familiar with the
-- internal structure of torch modules and the torch documentation.
-- You must complete the definitions of updateOutput and updateGradInput
-- for a 1-d log-exponential pooling module as explained in the handout.
--
-- Refer to the torch.nn documentation of nn.TemporalMaxPooling for an
-- explanation of parameters kW and dW.
--
-- Refer to the torch.nn documentation overview for explanations of the
-- structure of nn.Modules and what should be returned in self.output
-- and self.gradInput.
--
-- Don't worry about trying to write code that runs on the GPU.
--
-- Your submission should run on Mercer and contain:
-- a completed TEAMNAME_A3_skeleton.lua,
--
-- a script TEAMNAME_A3_baseline.lua that is just the provided A3_baseline.lua modified
-- to use your TemporalLogExpPooling module instead of nn.TemporalMaxPooling,
--
-- a saved trained model from TEAMNAME_A3_baseline.lua for which you have done some basic
-- hyperparameter tuning on the training data,
--
-- and a script TEAMNAME_A3_gradientcheck.lua that takes as input from stdin:
-- a float epsilon, an integer N, N strings, and N labels (integers 1-5)
-- and prints to stdout the ratios |(FD_epsilon_ijk - exact_ijk) / exact_ijk|
-- where exact_ijk is the backpropagated gradient for weight ijk for the given input
-- and FD_epsilon_ijk is the second-order finite difference of order epsilon
-- of weight ijk for the given input.
------------------------------------------------------------------------

local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)
    -----------------------------------------------
    -- assume input size is nbatch*inputsize(nrows)*nframe(ncols)
    local ndim = input:size():size()
    local nbatch = 0
    local inputsize = 0
    local nframe = 0

    -- reshape input to 3D with batch size as the first dimension.
    if ndim == 2 then
        nbatch = 1
        inputsize = input:size(1)
        nframe = input:size(2)
        input=input:reshape(nbatch,inputsize,nframe)
    elseif ndim == 3 then
        nbatch = input:size(1)
        inputsize = input:size(2)
        nframe = input:size(3)
    end

    --specify shape for output
    local outputsize = math.floor((inputsize-self.kW)/self.dW+1)
    self.output=torch.zeros(nbatch,outputsize,nframe)

    --Use for loops to update each output entry by looping over batch and pooling windows
    --loop over batch
    for nb = 1,nbatch do
        local outindex = 0
        --loop over pooling windows
        for ns = 1,inputsize-self.kW+1,self.dW do
            outindex = outindex+1
            --take a sample of input from the pooling window
            local insample = input[nb]:sub(ns,ns+self.kW-1):clone()
            --the sum needs to be written in a separate line for row-wise sum
            local insamplesize = insample:size(1)
            --exponentiate the sample
            insample:mul(self.beta):exp()
            --Eq: 1/beta*log(1/N*sum(exp(beta*sample[i])))
            local outsample = insample:sum(1):div(insamplesize):log():mul(1/self.beta)
            --update output on each input sample
            self.output[nb][outindex] = outsample
        end
    end
   -----------------------------------------------
   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   -----------------------------------------------
   -- assume input size is nbatch*inputsize(nrows)*nframe(ncols)
   local ndim = input:size():size()
   local nbatch = 0
   local inputsize = 0
   local nframe = 0

   -- reshape input to 3D with batch size as the first dimension.
   if ndim == 2 then
       nbatch = 1
       inputsize = input:size(1)
       nframe = input:size(2)
       input=input:reshape(nbatch,inputsize,nframe)
   elseif ndim == 3 then
       nbatch = input:size(1)
       inputsize = input:size(2)
       nframe = input:size(3)
   end

   -- specify shape for gradInput as the shape for input
   self.gradInput = torch.zeros(nbatch,inputsize,nframe)

   --Use for loops to update each gradinput entry by looping over batch and pooling windows
   --loop over batch
   for nb = 1,nbatch do
       local outindex = 0
       --loop over pooling windows
       for ns = 1,inputsize-self.kW+1,self.dW do
           outindex = outindex+1
           -- part 1: compute du/dx, element-wise exponential of input/sumof
           -- Eq: du/dx = exp(beta*sample[i])/sum(beta*sample)
           -- part 2: use the chain rule, dL/du * du/dx, dL/du = gradOutput

           --take a sample of input from the pooling window
           local insample = input[nb]:sub(ns,ns+self.kW-1):clone()
           --exponentiate the sample
           insample:mul(self.beta):exp()
           local expsample=insample:clone()
           --the sum needs to be written in a separate line for row-wise sum
           local insamplesum = insample:sum(1):clone()
           --update each entry for du/dx, loop over nframes
           local du_dx = torch.zeros(expsample:size())
           for idx = 1,du_dx:size(2) do
               --use :add to accumulate du/dx over pooling windows
               du_dx:sub(1,-1,idx,idx):add(expsample:sub(1,-1,idx,idx):div(insamplesum[1][idx])):mul(gradOutput[nb][outindex][idx])
           end
           --update each gradinput on batch, each pooling window
           self.gradInput[nb]:sub(ns,ns+self.kW-1):add(du_dx)
       end
   end
   -----------------------------------------------
   return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
