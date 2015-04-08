--------------------------------------------------------------------------------------
-- Baseline model
--------------------------------------------------------------------------------------

function model_baseline()
    -- construct model:
    model = nn.Sequential()
       
    -- Christian says: if you decide to just adapt the baseline code for part 2, you'll probably want to make this linear and remove pooling
    model:add(nn.TemporalConvolution(1, 20, 10, 1))
        
    --------------------------------------------------------------------------------------
    -- Replace this temporal max-pooling module with your log-exponential pooling module:
    --------------------------------------------------------------------------------------
    model:add(nn.TemporalMaxPooling(3, 1))
    
    model:add(nn.Reshape(20*39, true))
    model:add(nn.Linear(20*39, 5))
    model:add(nn.LogSoftMax())
    
    criterion = nn.ClassNLLCriterion()

    return model, criterion
end