'''
Created on Feb 28, 2017

@author: Michal.Busta at gmail.com
'''

import caffe
import os

  
def create_tiny_yolo_solver(args):
      
  solver = caffe.get_solver('models/tiny_solver.prototxt')
  #solver.restore('backup/yolo_mobile_iter_357000.solverstate')
  return solver

def create_recognizer_solver(args):
  solver = caffe.get_solver('models/solver_ctc.prototxt')
  #solver.restore('backup/recog_iter_195000.solverstate')
  return solver

  
def create_solvers_tiny(args):
    
  proposal_net = create_tiny_yolo_solver(args)
  recog = create_recognizer_solver(args)
  
  return proposal_net, recog

def create_models(buckets = [25, 50, 100], phase = caffe.TRAIN):
    
  transformers = create_spatial_transformers(buckets, phase)
  proposal_net = create_yolo(phase)
  recog = create_recognizer(phase)
  
  return proposal_net, transformers, recog


  
def create_models_tiny(phase = caffe.TEST):
  baseDir = os.path.dirname(os.path.abspath(__file__))
  proposal_net = caffe.Net('{0}/models/tiny.prototxt'.format(baseDir), '{0}/models/tiny.caffemodel'.format(baseDir), phase)
  recog = caffe.Net('{0}/models/model_cz.prototxt'.format(baseDir), '{0}/models/model.caffemodel'.format(baseDir), phase)
   
  return proposal_net, recog

    
