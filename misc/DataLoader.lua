local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)

  self.data_dir = opt.data_dir
  local tensor_vocab = path.join(self.data_dir, 'vocabulary.t7')
  local train_batches = path.join(self.data_dir, 'train.t7')
  local val_batches = path.join(self.data_dir, 'val.t7')
  local val_eval_batches = path.join(self.data_dir, 'val_eval.t7')

  if not path.exists(tensor_vocab) then
    local sents_train = path.join(self.data_dir, 'sents_train.txt')
    local sents_val = path.join(self.data_dir, 'sents_val.txt')
    local vocab_file = path.join(self.data_dir, 'vocabulary.txt')

    self:data_to_batches({sents_train, sents_val}, vocab_file, opt.data_dir, opt.batch_size, opt.max_sent_len)
  end

  -- load vocabulary
  local vocab_mappings = torch.load(tensor_vocab)
  self.word_to_ix = vocab_mappings[1]
  self.ix_to_word = vocab_mappings[2]
  self.vocab_size = 0
  for k,v in pairs(self.word_to_ix) do
    self.vocab_size = self.vocab_size+1
  end 

  self.iterators = {0, 0, 0}
  self.splits = {torch.load(train_batches), torch.load(val_batches), torch.load(val_eval_batches)}
end

function DataLoader:getBatch(split_ix)
  self.iterators[split_ix] = self.iterators[split_ix]+1
  if self.iterators[split_ix] > #self.splits[split_ix] then
    self.iterators[split_ix] = 1
  end

  return self.splits[split_ix][self.iterators[split_ix]][1], self.splits[split_ix][self.iterators[split_ix]][2]
end

function DataLoader:vocabSize()
  return self.vocab_size
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:splitSize(split_ix)
  return #self.splits[split_ix]
end

function DataLoader:resetIterator(split_ix)
  self.iterators[split_ix] = 0
end

function DataLoader:data_to_batches(sent_files, vocab_file, data_dir, batch_size, max_seq_len)

  local rawdata

  print('creating vocabulary mapping...')
  local word_to_ix = {}
  local ix_to_word = {}

  f = io.open(vocab_file)
  rawdata = f:read()
  local vocab_size = 0
  repeat 
    vocab_size = vocab_size+1
    word_to_ix[rawdata] = vocab_size
    ix_to_word[vocab_size] = rawdata
    rawdata = f:read()
  until not rawdata
  f:close()
  torch.save(path.join(data_dir, 'vocabulary.t7'), {word_to_ix, ix_to_word})

  local splits = {'train', 'val'}
  local split_batches = {}
  for split_ix=1,#splits do

    -- store all sentences as tensors
    local sents = {}
    local ids = {} -- store ids for val eval
    f = io.open(sent_files[split_ix])
    rawdata = f:read()
    repeat
      local sent = utils.split(rawdata, '%s')
      local sent_len = #sent
      if sent_len > max_seq_len then
        sent_len = max_seq_len
      end
      local sent_tensor = torch.Tensor(sent_len) -- #sent because sent[1] is the video id but add END token to tensor
      for ix=2,sent_len do
        sent_tensor[ix-1] = word_to_ix[sent[ix]]
      end
      sent_tensor[sent_len] = vocab_size+1 -- END TOKEN
      table.insert(sents, sent_tensor)
      table.insert(ids, sent[1])

      rawdata = f:read()
    until not rawdata

    -- create batches
    -- make even bathes
    local batches = {}
    if #sents % batch_size ~=0 then 
      repeat
        table.remove(sents, #sents)
      until #sents % batch_size == 0
    end

    local num_batches = #sents/batch_size
    local batch_perm = torch.randperm(#sents)
    for b=1,num_batches do
      -- find longest comment in batch
      longest_sent = 0
      for b_ix=1,batch_size do
        local ix = batch_perm[(b-1)*batch_size+b_ix]
        if sents[ix]:nElement() > longest_sent then
          longest_sent = sents[ix]:nElement()
        end
      end

      local batch_tensor = torch.Tensor(longest_sent, batch_size):zero()
      local batch_ids = {}
      for b_ix=1,batch_size do
        local ix = batch_perm[(b-1)*batch_size+b_ix]
        local sent = sents[ix]
        batch_tensor:select(2,b_ix):sub(1,sent:nElement()):copy(sent)
        table.insert(batch_ids, ids[ix])
      end
      table.insert(batches, {batch_tensor, batch_ids})
    end
    torch.save(path.join(data_dir, splits[split_ix]..'.t7'), batches)

    -- generate an extra val_eval file
    if split_ix==2 then
      local val_eval = {}
      local id_dict = {}
      for s=1,#sents do
        if id_dict[ids[s]] == nil then
          table.insert(val_eval, {sents[s], ids[s]})
          id_dict[ids[s]] = true
        end
      end
      print(#val_eval)
      torch.save(path.join(data_dir, 'val_eval.t7'), val_eval)
    end
  end
end