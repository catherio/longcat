--------------------------------------------------------------------------------------
-- Model based on Xiang Zhang's model
--------------------------------------------------------------------------------------

function get_model()
    -- construct model:
    model = nn.Sequential()
       
    -- Layer 1: From input, to convolution with pooling
    model:add(nn.TemporalConvolution(50, 128, 7, 1))
    	-- First layer:
	-- 	each word is a length-50 vector,
	-- 	we get 128 convolutional layers after this
	--	and each integrates over 7 words, with no stride
    model:add(nn.ReLU())
    model:add(nn.TemporalMaxPooling(3, 3))
    	-- Inspired by Text Understanding from Scratch, but starting
	-- at their 'layer 2' because we already have words, we now
	-- pool at a size of 3

	-- If our input was length 198, it went down to 192, then pooled to 64

    -- Layers 2 and 3: Convolutional layers
    model:add(nn.TemporalConvolution(128, 128, 3, 1))
    model:add(nn.ReLU())

    model:add(nn.TemporalConvolution(128, 128, 3, 1))
    model:add(nn.ReLU())

	-- Input length should now be 60

    model:add(nn.TemporalMaxPooling(3, 3))
    	-- Then pool to 20

    -- Layer 4: Fully connected, with dropout
    model:add(nn.Reshape(20*128, true))
    model:add(nn.Dropout(0.5))

    model:add(nn.Linear(20*128, 128))
    model:add(nn.ReLU())

    -- Output layer
    model:add(nn.Linear(128, 5))
    model:add(nn.LogSoftMax())
    
    criterion = nn.ClassNLLCriterion()

    return model, criterion
end