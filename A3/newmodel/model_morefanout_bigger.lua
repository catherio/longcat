--------------------------------------------------------------------------------------
-- Model based on Xiang Zhang's model
--------------------------------------------------------------------------------------

function get_model()
    -- construct model:
    model = nn.Sequential()
       
    -- Layer 1: From input, to convolution
    model:add(nn.TemporalConvolution(100, 256, 7, 1))
    	-- First layer:
	-- 	each word is a glove vector, size is in the first dimension
	-- 	we get 256 convolutional filters after this
	--	and each integrates over 7 words, with no stride
    model:add(nn.ReLU())
	-- If our input was length 104, it went down to 98

    -- Layers 2: Convolutional layer with pooling
    model:add(nn.TemporalConvolution(256, 128, 3, 1))
    model:add(nn.ReLU())
    model:add(nn.TemporalMaxPooling(2, 2))
	-- 98 -> 96 -> 48

    -- Layer 3: Fully connected, with dropout
    model:add(nn.Reshape(48*128, true))
    model:add(nn.Dropout(0.5))

    model:add(nn.Linear(48*128, 128))
    model:add(nn.ReLU())

    -- Output layer
    model:add(nn.Linear(128, 5))
    model:add(nn.LogSoftMax())
    
    criterion = nn.ClassNLLCriterion()

    return model, criterion
end