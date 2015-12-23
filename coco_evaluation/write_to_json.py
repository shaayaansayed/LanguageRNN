#!/usr/bin/env python

import io
import json
import sys
import getopt
import hashlib

class CocoResFormat:
  def __init__(self):
    self.res = []
    self.caption_dict = {}

  def read_multiple_files(self, filelist,hash_img_name):
    for filename in filelist:
      print 'In file %s\n' % filename
      self.read_file(filename,hash_img_name)

  def read_file(self,filename,hash_img_name):
    count = 0
    with open(filename,'r') as opfd:
      for line in opfd:
        count +=1
        id_sent = line.strip().split('\t')
        if len(id_sent)>2:
          id_sent = id_sent[-2:]
        assert len(id_sent) == 2
        sent = id_sent[1].decode('ascii', 'ignore')

        if hash_img_name:
          img_id = int(int(hashlib.sha256(id_sent[0]).hexdigest(), 16) % sys.maxint)
        else:  
          img = id_sent[0].split('_')[-1].split('.')[0]
          img_id = int(img)
        imgid_sent = {}
        
        if img_id in self.caption_dict:
          assert self.caption_dict[img_id] == sent
        else:
          self.caption_dict[img_id] = sent
          imgid_sent['image_id'] = img_id
          imgid_sent['caption'] = sent
          self.res.append(imgid_sent)
        if count%1000 == 0:
          print 'Processed %d ...' % count

  def dump_json(self, outfile):
    res = self.res
    # json.dump(res, open(outfile, 'w'))
    with io.open(outfile, 'w', encoding='utf-8') as fd:
      fd.write(unicode(json.dumps(res, ensure_ascii=False,sort_keys=True,indent=2,separators=(',', ': '))))

def main(scriptname, input_file,  argv):
  
  hash_img_name = False
  usage = '%s <input_file> -x (if you want to hash the image id)' % scriptname
  default_values = '%s <input_file> ' % (scriptname)
  
  try:
    opts, args = getopt.getopt(argv,'hx')
  except getopt.GetoptError:
    print "Usage:"
    print usage
    print "Default values:"
    print default_values
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print "Usage:"
      print usage
      print "Default values:"
      print default_values
      sys.exit()
    if opt == '-x':
      hash_img_name = True
  
    # elif opt in ('-i'):
    #   model_tag = arg
    # elif opt in ('-o'):
    #   setting = arg
  
  output_file = '{0}.json'.format(input_file)
  crf = CocoResFormat()
  crf.read_file(input_file,hash_img_name)
  crf.dump_json(output_file)
  print output_file
  
if __name__ == "__main__":
  main(sys.argv[0], sys.argv[1], sys.argv[2:])

