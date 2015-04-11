require 'torch'

ffi = require('ffi')

function preprocess_data(raw_data, wordvector_table, opt)
    
    local data = torch.zeros(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), opt.inputDim, 1)
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
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            
            -- break each review into words and compute the document average
            for word in document:gmatch("%S+") do
                if wordvector_table[word:gsub("%p+", "")] then
                    doc_size = doc_size + 1
                    data[k]:add(wordvector_table[word:gsub("%p+", "")])
                end
            end

            data[k]:div(doc_size)
            labels[k] = i
        end
    end

    return data, labels
end