--------------------------------------------------------------------------------------
-- Model based on Xiang Zhang's model
--------------------------------------------------------------------------------------

function get_model()
    -- construct model:
    model = nn.Sequential()
       
    -- Layer 1: From input, to convolution
    model:add(nn.TemporalConvolution(50, 256, 7, 1))
    	-- First layer:
	-- 	each word is a length-50 vector,
	-- 	we get 128 convolutional layers after this
	--	and each integrates over 7 words, with no stride
    model:add(nn.ReLU())
	-- If our input was length 198, it went down to 192
	-- Try no pooling yet!

    -- Layers 2 and 3: Convolutional layers with pooling
    model:add(nn.TemporalConvolution(256, 256, 3, 1))
    model:add(nn.ReLU())
    model:add(nn.TemporalMaxPooling(2, 2))
	-- 192 -> 190 -> 95

    model:add(nn.TemporalConvolution(256, 128, 3, 1))
    model:add(nn.ReLU())
    model:add(nn.TemporalMaxPooling(3, 3))
    	-- 95 -> 93 -> 31

    -- Layer 4: Fully connected, with dropout
    model:add(nn.Reshape(31*128, true))
    model:add(nn.Dropout(0.5))

    model:add(nn.Linear(31*128, 128))
    model:add(nn.ReLU())

    -- Output layer
    model:add(nn.Linear(128, 5))
    model:add(nn.LogSoftMax())
    
    criterion = nn.ClassNLLCriterion()

    return model, criterion
end