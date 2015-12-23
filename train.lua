require 'torch'
require 'nn'
require 'nngraph'

-- local inputs
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.DataLoader'
require 'misc.LanguageModel'
require 'misc.optim_updates'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

cmd:option('-data_dir','data/lm','directory where data is')
cmd:option('-checkpoint_dir','cv/','directory where checkpoints and language evals are saved')
cmd:option('-save_checkpoint_every',200, 'number of iterations to save a model checkpoint')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
-- Model settings
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',2000,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-batch_size',16,'number of instances in a training batch')
cmd:option('-num_layers',2,'number of layers in lstm')
cmd:option('-max_sent_len',80,'maximum sequence length allowed -- includes END token')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-enc',-1,'is this an encoding model?')
-- Optimization 
cmd:option('-epochs', -1, 'max number of epochs to run for (-1 = run forever)')
cmd:option('-optim','rmsprop','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-learning_rate_decay_start', 10, 'at what epoch to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 1, 'every how many epochs thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
-- Misc
cmd:option('-id', 1, 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')

cmd:text()

-- basic torch initializations
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-- create data loader 
local loader = DataLoader{data_dir=opt.data_dir, batch_size=opt.batch_size, max_sent_len=opt.max_sent_len}
utils.setVocab(loader:getVocab())

-- initialize the network
lmOpt = {}
lmOpt.vocab_size = loader:vocabSize()
lmOpt.input_encoding_size = opt.input_encoding_size
lmOpt.max_sent_len = opt.max_sent_len
lmOpt.num_layers = opt.num_layers
lmOpt.rnn_size = opt.rnn_size
lmOpt.batch_size = opt.batch_size

protos = {}
protos.lm = nn.LanguageModel(lmOpt)
protos.crit = nn.LanguageModelCriterion(lmOpt)

-- send the model to cuda 
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

local params, grad_params = protos.lm:getParameters()
print('total number of parameters in LM: ', params:nElement())

print('creating thin models for checkpointing...')
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias') 
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end -- sets gradient weights to nil to reduce memory

protos.lm:createClones()

collectgarbage()

-- validation evaluation
local function eval_split(split)

  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split

  local loss_sum = 0
  local loss_evals = #loader.splits[2]
  local vocab = loader:getVocab()
  local samples = {}

  for i=1,loss_evals do

    -- fetch a batch of data
 	local batch_sent, ids = loader:getBatch(split)

    if opt.gpuid >= 0 then
      batch_sent = batch_sent:float():cuda()
    end

    -- forward the model to get loss
    local logprobs = protos.lm:forward{batch_sent}
    local loss = protos.crit:forward(logprobs, batch_sent)
    loss_sum = loss_sum + loss

    print(i .. '/' .. loss_evals .. '...')
  end

  return loss_sum/loss_evals
end

-- language evaluation by sampling from the network
local function sample_split(split_ix)
	
	protos.lm:evaluate()
	loader:resetIterator(split_ix)

  local num_samples = loader:splitSize(split_ix)

  local split_sample
  local vidIds = {}
  for ix=1,num_samples do

    local sent, id = loader:getBatch(3)

    local primetext
    -- if sampling one sentence at a time
    if sent:nDimension() == 1 then
      primetext = sent:sub(1, 1)
    else
      primetext = sent[1]
    end

    if opt.gpuid >= 0 then
      primetext = primetext:float():cuda()
    end

    sample, logprobs = protos.lm:sample(primetext, nil)

    if opt.gpuid >= 0 then
      sample = sample:float():cuda()
    end
    -- concatenate sample with primetext and the rest of the samples in the split
    if ix == 1 then
      split_sample = sample
    else
      split_sample = torch.cat(split_sample, sample, 2)
    end
    table.insert(vidIds, id)
  end
  return split_sample, vidIds
end

-- loss function
local function lossFun()
  protos.lm:training()
  grad_params:zero()

  -- get batch of data  
  local batch_sent, _ = loader:getBatch(1)

  if opt.gpuid >= 0 then
    batch_sent = batch_sent:float():cuda()
  end
  
  -- forward pass
  local logprobs = protos.lm:forward{batch_sent}
  local loss = protos.crit:forward(logprobs, batch_sent)

  -- backward pass
  local dlogprobs = protos.crit:backward(logprobs, batch_sent)
  local ddumpy= protos.lm:backward(batch_sent, dlogprobs)

  -- clip gradients
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  local losses = { total_loss = loss }
  return losses
end

local ix_to_word = loader:getVocab()
local loss0
local iter = 0
local optim_state = {}
local ntrain = loader:splitSize(1)
local best_score = 0
while true do
  iter = iter + 1

  -- eval loss/gradient 
  local epoch = iter / ntrain
  local losses = lossFun()

  -- decay learning rate 
  local learning_rate = opt.learning_rate
  if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local epochs_over_start = math.ceil(epoch - opt.learning_rate_decay_start)
    local decay_factor = math.pow(opt.decay_rate, epochs_over_start)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end

  -- optimization step
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- save checkpoint based on language evaluation
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
  	
    local sample_tensor, ids = sample_split(3)
    score, samples = utils.lang_eval(sample_tensor, ids, 'METEOR')

    -- save the model if it performs better than ever
    if score > best_score then
      local checkpoint_path = path.join(opt.checkpoint_dir, 'id_' .. opt.id)

      local checkpoint_info = {}
      checkpoint_info.opt = opt
      checkpoint_info.epoch = epoch
      checkpoint_info.vocab = ix_to_word
      checkpoint_info.score = score
      checkpoint_info.samples = samples
      utils.write_json(checkpoint_path .. '.json', checkpoint_info)

      local save_protos = {}
      save_protos.lm = thin_lm
      torch.save(checkpoint_path .. '.t7', save_protos)

      best_score = score
    end
  end

  if iter % opt.print_every == 0 then
    print(string.format("%d (epoch %.3f), train_loss = %6.8f", iter, epoch, losses.total_loss))
  end

  if iter == opt.max_iters then
    break
  end

end