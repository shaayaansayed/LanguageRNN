#!/usr/bin/env python
# encoding: utf-8


import sys
import os
import getopt
import io
import json

sys.path.append('./coco-caption/')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')




def main(scriptname, prediction_file, annotation_file, argv):
  
  hash_img_name = False
  usage = '%s <prediction_file> <annotation_file> -x (if you want to hash the image id)' % scriptname
  default_values = '%s <prediction_file> <annotation_file> ' % (scriptname)
  
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
  
  #output_file = '{0}.json'.format(input_file)
  #crf = CocoResFormat()
  #crf.read_file(input_file,hash_img_name)
  #crf.dump_json(output_file)
  
  
   
  # create coco object and cocoRes object
  coco = COCO(annotation_file)
  cocoRes = coco.loadRes(prediction_file)
  
  # In[4]:
  
  # create cocoEval object by taking coco and cocoRes
  cocoEval = COCOEvalCap(coco, cocoRes)
  
  # evaluate on a subset of images by setting
  # cocoEval.params['image_id'] = cocoRes.getImgIds()
  # please remove this line when evaluating the full validation set
  #cocoEval.params['image_id'] = cocoRes.getImgIds()
  
  # evaluate results
  cocoEval.evaluate()
  
  
  # In[5]:
  
  scores = {}  
  # print output evaluation scores
  for metric, score in cocoEval.eval.items():
    scores[metric] = score
    print '%s: %.3f'%(metric, score)
  
  outfile = "{0}.json".format(prediction_file.split(".txt")[0])
  with io.open(outfile, 'w', encoding='utf-8') as fd:
    fd.write(unicode(json.dumps(scores)))
 
if __name__ == "__main__":
  main(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3:])

