require 'torch'

ffi = require('ffi')

function preprocess_data(raw_data, wordvector_table, opt)

    maxWords = 104

    -- Anticipate how many extra features there will be
    extraFeat = 1; -- just the punctuation feature

    -- Dimensions here are nDocs * docLength * featureSize
    local data = torch.zeros(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), maxWords, opt.inputDim + extraFeat)
    local labels = torch.zeros(opt.nClasses*(opt.nTrainDocs + opt.nTestDocs))
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    local order = torch.randperm(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))

    for i=1,opt.nClasses do
        for j=1,opt.nTrainDocs+opt.nTestDocs do
            local k = order[(i-1)*(opt.nTrainDocs+opt.nTestDocs) + j]
	    	  -- the first i/nClasses sized chunk of "order" is interpreted as the indices
		  -- in the final matrix where the ith class will be put

            local doc_size = 1
            
            local index = raw_data.index[i][j]

            -- load document and standardize
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower() --lowercase
            document = document:gsub("\\n", " ") --replace newlines with spaces
	    document = document:gsub("(%p+)", " %1 ") --wrap punctuation with spaces

            -- break each review into words and put them in the table
	    wordsSoFar = 0
            for word in document:gmatch("%S+") do
	    	if wordsSoFar >= maxWords then
		   break
		end

		if word:gmatch("%p+") ~= "" then -- punctuation
		   wordsSoFar = wordsSoFar + 1
		   data[k][wordsSoFar][-1] = 1 -- set the "punctuation" flag
		else if wordvector_table[word:gsub("%p+", "")] then --non-punctuation
		    wordsSoFar = wordsSoFar + 1
		    wordvec = wordvector_table[word:gsub("%p+", "")])
                    data[k][wordsSoFar][{{1,-2}}]:add(wordvec) -- fill up to the second-to-last bin; space for punctuation
                end
            end
	    -- If we run out of words, don't do anything fancy; just leave the rest zeros

            labels[k] = i
        end
    end

    return data, labels
end