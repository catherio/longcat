require 'io'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'


--- Parses and loads the GloVe word vectors into a hash table:
-- glove_table['word'] = vector
function load_glove(path, inputDim)
    local glove_file = io.open(path)
    local glove_table = {}
    local line = glove_file:read("*l")
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    glove_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end

    return glove_table
end


function preprocess_data(msg, wordvector_table, inputDim)
    maxWords = 104
    -- Dimensions here are nDocs * docLength * featureSize
    local data = torch.zeros(1, maxWords, inputDim)
    local doc_size = 1
    -- load document and standardize
    local document = msg:lower() --lowercase
    document = document:gsub("\\n", " ") --replace newlines with spaces

    -- break each review into words and put them in the table
    wordsSoFar = 0
    for word in document:gmatch("%S+") do
        if wordsSoFar >= maxWords then
            break
        end

        if wordvector_table[word:gsub("%p+", "")] then
            wordsSoFar = wordsSoFar + 1
            data[1][wordsSoFar]:add(wordvector_table[word:gsub("%p+", "")])
        end
    end

    -- If we run out of words, don't do anything fancy; just leave the rest zeros
    return data
end


-- Read the number of strings, n, to read fron stdin
-- then iterate over each and return the class label
-- Note that io.read() does not return '\n', so our
-- model should take that into account if necessary
function main()
    --load glove table
    print("Loading word vectors...")
    inputDim = 100
    glovePath = "/scratch/courses/DSGA1008/A3/glove/glove.6B." .. inputDim .. "d.txt" -- path to raw glove data .txt file
    glove_table = load_glove(glovePath, inputDim)

    print("Loading trained model...")
    --load model
    modelPath = "/scratch/ls3470/DeepLearning/longcat/A3/model.net"
    model = torch.load(modelPath)
    model:cuda()
    model:evaluate()

    print("Total number of reviews: ")
    n = io.read()
    for i=1,n
    do
      local msg = io.read()
      data = preprocess_data(msg, glove_table, inputDim)
      data = data:cuda()
      pred = model:forward(data)
      m,ptarget = pred:max(2)
      io.write(ptarget[1][1],'\n')

      -- e.g. class = model_output
      -- io.write( class, '\n' )
      -- io.write('we all love ', msg, '\n')
    end
end

main()
