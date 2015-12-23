# convert results to json format

import io
import json

class CocoResFormat:
  def __init__(self):
    self.res = []

  def read_multiple_files(self, filelist):
    for filename in filelist:
      print 'In file %s\n' % filename
      self.read_file(filename)

  def read_file(self,filename):
    count = 0
    with open(filename,'r') as opfd:
      for line in opfd:
        count +=1
        id_sent = line.strip().split('\t')
        assert len(id_sent)==2
        sent = id_sent[1]
        img = id_sent[0].split('_')[-1].split('.')[0]
        img_id = int(img)
        imgid_sent = {}
        imgid_sent['image_id'] = img_id
        imgid_sent['caption'] = sent
        self.res.append(imgid_sent)
        if count%100 == 0:
          print 'Processed %d ...' % count

  def dump_json(self, outfile):
    res = self.res
    # json.dump(res, open(outfile, 'w'))
    with io.open(outfile, 'w', encoding='utf-8') as fd:
      fd.write(unicode(json.dumps(res, ensure_ascii=False)))

def main():
  NNINPUT = 'val' #'test' #'val'
  NNDATA = 'trainval' #'train'
  PREFIX = './results/coco2014_fc7_all_full_%s_on_%s_s2s.%s'
  data_type = 'val2014'
  # data_type = 'test2014'
  algo_name = '1nn'
  OUTPREFIX = \
  './results/captions_%s_s2s_results.json'
  TXT = 'txt'
  JSON = 'json'
  mat_res_file = PREFIX % (NNINPUT, NNDATA, TXT)
  output_file = OUTPREFIX % (data_type)
  crf = CocoResFormat()
  crf.read_file(mat_res_file)
  crf.dump_json(output_file)

if __name__=="__main__":
  main()


  #NNINPUT = 'mytest'
  #NNINPUT2 = 'vallstm'
  # PREFIX = '/home/vsubhashini/Mooney/lstm/nnbase/coco2014_fc7_%s_on_%s_knntest.%s'
  # mat_mytest_file = PREFIX % (NNINPUT, NNDATA, TXT)
  # mat_vallstm_file = PREFIX % (NNINPUT2, NNDATA, TXT)
  # crf.read_multiple_files([mat_mytest_file, mat_vallstm_file])
