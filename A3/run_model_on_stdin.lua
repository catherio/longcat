require 'io'

-- Read the number of strings, n, to read fron stdin
-- then iterate over each and return the class label
-- Note that io.read() does not return '\n', so our
-- model should take that into account if necessary

n = io.read()
for i=1,n
do
  local msg = io.read()
  -- ADD MODEL OUTPUT HERE INSTEAD OF adding WE ALL LOVE
  -- it should spit out the class label, 
  -- e.g. class = model_output
  -- io.write( class, '\n' )
  io.write('we all love ', msg, '\n') 
end
