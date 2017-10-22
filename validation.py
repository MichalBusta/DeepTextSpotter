# -*- coding: utf-8 -*-

import sys, os

sys.path.insert(0, '/home/busta/git/caffe/python')

import caffe
import cv2

import math, time

from data import DataLoader

import vis
import matplotlib.pyplot as plt
import argparse
import numpy as np

from models import create_models_tiny
from data import DataLoader
from utils import intersect, union, area, print_seq, get_normalized_image, print_seq2, print_seq_ext, get_obox, process_splits
import cmp_trie

caffe.set_mode_gpu()
caffe.set_device(0)  # if we have multiple GPUs, pick the first one

'''Start declaring global parameters.'''

image_no = 0
buckets = [54, 80, 124, 182, 272, 410, 614, 922, 1383, 2212]      
image_size = [544, 544] # Size of testing image
it = 0
mean_rec = 0
gt_all = 0
gt_loc = 0
wr_good_all = 0
debug = True # Debug parameter
rec_t = 0.6
ext_factor = 1.2
ext_factorx = 1.2
use_per_image = False
det_count = 0

'''End declaring global parameters.'''
    
to_cls_x = []
to_cls_y = []
               
def evaluate_image(batch, detections, word_gto, iou_th=0.3, iou_th_vis=0.5, iou_th_eval=0.4):
    
  '''
  Summary : Returns end-to-end true-positives, detection true-positives, number of GT to be considered for eval (len > 2).
  Description : For each predicted bounding-box, comparision is made with each GT entry. Values of number of end-to-end true
                positives, number of detection true positives, number of GT entries to be considered for evaluation are computed.
  
  Parameters
  ----------
  iou_th_eval : float
      Threshold value of intersection-over-union used for evaluation of predicted bounding-boxes
  iou_th_vis : float
      Threshold value of intersection-over-union used for visualization when transciption is true but IoU is lesser.
  iou_th : float
      Threshold value of intersection-over-union between GT and prediction.
  word_gto : list of lists
      List of ground-truth bounding boxes along with transcription.
  batch : list of lists
      List containing data (input image, image file name, ground truth).
  detections : tuple of tuples
      Tuple of predicted bounding boxes along with transcriptions and text/no-text score.
  
  Returns
  -------
  tp : int
      Number of predicted bounding-boxes having IoU with GT greater than iou_th_eval.
  tp_e2e : int
      Number of predicted bounding-boxes having same transciption as GT and len > 2.
  gt_e2e : int
      Number of GT entries for which transcription len > 2.
  '''
  
  gt_to_detection = {}
  tp = 0
  tp_e2e = 0
  gt_e2e = 0
  
  draw = batch[4][0]    
  normFactor = math.sqrt(draw.shape[1] * draw.shape[1] + draw.shape[0] * draw.shape[0]) # Normalization factor
  for i in range(0, len(detections)):
      
    det = detections[i]
    boxr = det[0]
    box = cv2.boxPoints(boxr) # Predicted bounding-box parameters
    box = np.array(box, dtype="int") # Convert predicted bounding-box to numpy array
    bbox = cv2.boundingRect(box)
    
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    bbox[2] += bbox[0] # Convert width to right-coordinate
    bbox[3] += bbox[1] # Convert height to bottom-coordinate
    
    vis.draw_box_points(draw, box, color = (255, 0, 0))
    
    det_text = det[1][0] # Predicted transcription for bounding-box
    #print(det_text)
    
    for gt_no in range(len(word_gto)):
        
      gt = word_gto[gt_no]
      txt = gt[5] # GT transcription for given GT bounding-box
      gtbox  = ((gt[0] * draw.shape[1], gt[1] * draw.shape[0]), (gt[2] * normFactor, gt[3] * normFactor), gt[4] * 180 / 3.14) # Re-scaling GT values
      gtbox = cv2.boxPoints(gtbox)
      gtbox = np.array(gtbox, dtype="int")
      rect_gt = cv2.boundingRect(gtbox)
      
      
      rect_gt = [rect_gt[0], rect_gt[1], rect_gt[2], rect_gt[3]]
      rect_gt[2] += rect_gt[0] # Convert GT width to right-coordinate
      rect_gt[3] += rect_gt[1] # Convert GT height to bottom-coordinate 

      inter = intersect(bbox, rect_gt) # Intersection of predicted and GT bounding-boxes
      uni = union(bbox, rect_gt) # Union of predicted and GT bounding-boxes
      ratio = area(inter) / float(area(uni)) # IoU measure between predicted and GT bounding-boxes
      
      # 1). Visualize the predicted-bounding box if IoU with GT is higher than IoU threshold (iou_th) (Always required)
      # 2). Visualize the predicted-bounding box if transcription matches the GT and condition 1. holds
      # 3). Visualize the predicted-bounding box if transcription matches and IoU with GT is less than iou_th_vis and 1. and 2. hold
      if ratio > iou_th:
        vis.draw_box_points(draw, box, color = (0, 128, 0))
        if not gt_to_detection.has_key(gt_no):
          gt_to_detection[gt_no] = [0, 0]
            
        if txt.lower() == det_text.lower():
          to_cls_x.append([len(det_text), det[1][1], det[1][2], det[1][3]])
          to_cls_y.append(1)
          vis.draw_box_points(draw, box, color = (0, 255, 0), thickness=2)
          gt[7] = 1 # Change this parameter to 1 when predicted transcription is correct.
          
          if ratio < iou_th_vis:
              vis.draw_box_points(draw, box, color = (255, 255, 255), thickness=2)
              cv2.imshow('draw', draw) 
              #cv2.waitKey(0)
                
        else:
          to_cls_x.append([len(det_text), det[1][1], det[1][2], det[1][3]])
          to_cls_y.append(0)
          
        tupl = gt_to_detection[gt_no] 
        if tupl[0] < ratio:
          tupl[0] = ratio 
          tupl[1] = i   
                  
  # Count the number of end-to-end and detection true-positives
  for gt_no in range(len(word_gto)):
    gt = word_gto[gt_no]
    txt = gt[5]
    if len(txt) > 2:
      gt_e2e += 1
      if gt[7] == 1:
        tp_e2e += 1
            
    if gt_to_detection.has_key(gt_no):
      tupl = gt_to_detection[gt_no] 
      if tupl[0] > iou_th_eval: # Increment detection true-positive, if IoU is greater than iou_th_eval
        tp += 1             
          
  cv2.imshow('draw', draw)             
  return tp, tp_e2e, gt_e2e 
  
  
def ocr_detections(net_ctc, img, scaled_img, boxes, image_size, r_p_th, out_raw, baseName, debug, split_words, alow_non_dict=False):
    
  global rec_t, ext_factor, use_per_image
    
  draw = np.copy(scaled_img)
    
  # Region layer returns normalized coordiates, convert the generated boxes to image coordinate system
  boxes[0, 0, :, 0] *= image_size[0]
  boxes[0, 0, :, 1] *= image_size[1]
  normFactor = math.sqrt(image_size[1] * image_size[1] + image_size[0] * image_size[0])
  boxes[0, 0, :, 2] *= normFactor
  boxes[0, 0, :, 3] *= normFactor
  
  nms_mask = boxes[0, 0, :, 8] != 1
  boxes = boxes[:, :, nms_mask, :]
  
    # Region layer returns boxes in sorted order by r_{p}, filter out the boxes with r_{p} below threshold value
  boxes_count = 0
  for i in range(0, boxes.shape[2]):
      det_word = boxes[0, 0, i]
      if (det_word[0] == 0 and det_word[1] == 0) or det_word[5] < r_p_th:
        break
      boxes_count += 1
  
  detections_out = []
  
  for i in range(0, boxes_count):
      
    det_word = boxes[0, 0, i]
    boxr  = ((det_word[0], det_word[1]), (det_word[2], det_word[3]), det_word[4] * 180 / 3.14) # Convert the rotation parameter to degrees
    box = cv2.boxPoints(boxr) # Gives the coordinates for 4 points of bounding-box
    box = np.array(box, dtype="int")
    
    if det_word[3] < 5:
      continue
    
    if debug:
      try:
        vis.draw_box_points(draw, box, (255, 0, 0)) # Visualize the predicted bounding-boxes
      except:
        pass
    
    bbox = cv2.boundingRect(box)
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    bbox[2] += bbox[0] # Convert width to right-coordinate
    bbox[3] += bbox[1] # Convert height to bottom-coordinate
    
    boxro  = [[det_word[0], det_word[1]], [det_word[2]  * ext_factorx, det_word[3] * ext_factor], det_word[4] * 180 / 3.14] # Re-scaling the bounding-box parameters to increase height and width, this helps recognizer
    boxt = get_obox(scaled_img, img, boxro) # Rescale the predicted bounding box to original image size
    boxt = ((boxt[0][0], boxt[0][1]), (boxt[1][0], boxt[1][1]), boxt[2])
    
    norm2, rot_mat = get_normalized_image(img, boxt) # norm2 stores normalized cropped region from original image determined by predicted bounding box
    if norm2 is None:
      continue
    #boxt[2] = boxt[2] * 180 / 3.14
    #cv2.imshow('norm2', norm2)
    #cv2.imshow('draw', draw)
    if norm2.ndim > 2:
        norm = cv2.cvtColor(norm2, cv2.COLOR_BGR2GRAY ) # Convert the cropped region to GRAY scale for recognizer
    else:
        norm = norm2 # Do nothing if already GRAY scale                             
    
    # Change width for each cropped region, keeping height fixed (32). Map width to closest value from bucket
    width_scale = 32.0 / norm2.shape[0]
    width = norm.shape[1] * width_scale
    best_diff = width
    bestb = 0
    for idx, val in enumerate(buckets):
      if (buckets[idx] - width) < 0  :
          bestb = idx
          best_diff = abs(buckets[idx] - width) * 3
          continue
      if best_diff > (buckets[idx] - width): 
          bestb = idx
          best_diff = (buckets[idx] - width)
    scaled = cv2.resize(norm, (buckets[bestb], 32)) # Resize cropped region for input for recognizer FCN
         
    if scaled.ndim == 3:
      scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY) 
    
    imtf = np.asarray([scaled], dtype=np.float)
    imtf = np.asarray(imtf, dtype=np.float)
    imtf /= 128.0
    imtf -= 1
    imtf = np.reshape(imtf, (imtf.shape[0], -1, imtf.shape[1], imtf.shape[2])) 
        
    net_ctc.blobs['data'].reshape(imtf.shape[0],imtf.shape[1],imtf.shape[2], imtf.shape[3]) # Reshape the recognizer FCN to adapt varying cropped region size
    net_ctc.blobs['data'].data[...] = imtf # Load the data onto recognizer FCN (cropped region data)
    net_ctc.forward() # Recognizer FCN feed-forward
    ctc_f = net_ctc.blobs['softmax'].data[...] 
    
    ctc_f = ctc_f.reshape(ctc_f.shape[0], ctc_f.shape[1], ctc_f.shape[3])
    labels = ctc_f.argmax(2) # 3rd dimension (ctc_f[:,:,2]) contains softmax distribution over all the possible characters for each position, thus labels store the index of character with maximum value (probability).
    mask = labels > 3
    masked = ctc_f.max(2)[mask] # For each predicted character, fetch the corresponding score
    mean_conf = np.sum(masked) / masked.shape[0] # Mean score for all the predicted characters
    
        # Visualize if mean score for predicted characters is less than 0.3
    if mean_conf < 0.3:
      continue
    
    if debug:    
      vis.vis_square(imtf[0])
    
    det_text, conf, dec_s = print_seq_ext(labels[:, 0], np.sum(masked) ) 
    if not split_words:
      detections_out.extend( [(boxt, (det_text, mean_conf, 1, mean_conf) )] )
      continue
    
    #print(det_text)
    #str_lm, pr =  cmp_trie.decode_sofmax_lm(ctc_f.reshape(ctc_f.shape[0], ctc_f.shape[2]))
    #if det_text != str_lm:
    #  print('  Decoding diff: {0} - {1}'.format(det_text, str_lm))
    #  det_text = str_lm.strip()
    
    if len(det_text.strip()) == 0:
      continue
    
    if len(det_text.strip()) <= 3:
      if mean_conf < 0.6 or det_word[5] < 0.4:
        continue
    
    pr = 1
    for k in range(masked.shape[0]):
      pr = pr *  masked[k]
    pr = math.exp(pr)
    #pr = math.pow(pr, 1.0/ len(det_text) )
    
    #tex_conf =  mean_conf / ctc_f.shape[0]
    #if tex_conf < 0.1:
    #  continue
    
    #print(det_text)
    #cv2.imshow('norm2', norm2)
    splits_raw = process_splits(det_text, conf, dec_s, norm2, ctc_f, rot_mat, boxt, img, det_word[5], mean_conf, alow_non_dict = alow_non_dict) # Process the split and improve the localization results using "space" (' ') predicted by recognizer
    detections_out.extend( splits_raw )
    spl = det_text.split(" ")
    
    if len(spl) == 1 and cmp_trie.is_dict(spl[0].lower().encode('utf-8')) == 1:
      continue
                  
    
    dec2, conf2, dec_splits = cmp_trie.decode_sofmax(ctc_f.reshape(ctc_f.shape[0], ctc_f.shape[2]))
    best_dict = print_seq2(dec2[0])
    
    if out_raw is not None and len(det_text) > 2:
      boxout = cv2.boxPoints(boxt)    
      out_raw.write(u"{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}\n".format(\
              baseName[:-4],boxout[0, 0],boxout[0, 1], boxout[1, 0], boxout[1, 1], \
              boxout[2, 0], boxout[2, 1], boxout[3, 0], boxout[3, 1], det_text, best_dict, mean_conf).encode('utf8'))
  
    splits_out = process_splits(best_dict, conf2, dec_splits, norm2, ctc_f, rot_mat, boxt, img, det_word[5], pr, alow_non_dict=False)
    detections_out.extend( splits_out )
  
  #detections_out = nms(detections_out)
  if out_raw is not None:
    out_raw.flush()   
  
  cv2.imshow('draw', draw)
  cv2.waitKey(10)  
  return detections_out 
    
def forward_image(nets, batch, image_size, debug = True, write_results=False, out_raw = None, r_p_th=0.1, split_words=True, alow_non_dict=False):

  '''
  Summary : Takes the trained localization, recognition networks and raw image as input, returns predicted bounding-boxes and transcriptions.
  Description : 1). For given input image, the FCN is reshaped to adapt to image size. 
                2). Whole image is used for feed-forward operation in FCN and anchor boxes are generated. 
                3). For each anchor box generated from FCN, recognizer FCN is reshaped to adapt to size.

  Parameters
  ----------
  r_p_th : float
     Float which represents threshold value for filtering generated output from FCN. (r_{p} in paper).
  out_raw : 
  write_results : boolean
  debug : boolean
     Boolean parameter representing debug mode, it visualizes the generated output if set True.
  image_size : list
     List of len = 2, contains the dimension of testing image.
  batch : list of lists
     List containing data (input image, image file name, ground truth). 
  nets : list with two caffe nets
     List which contains text-localization net and text-recognition networks.

  Returns
  -------
  tp : int
     Number of predicted bounding-boxes having IoU with GT greater than iou_th_eval.
  tp_e2e : int
     Number of predicted bounding-boxes having same transciption as GT and len > 2.
  gt_e2e : int
     Number of GT entries for which transcription len > 2.
  detections_out : List of tuples
     List of tuples containing predicted bounding boxes along with transcriptions and text/no-text score.
  '''
  
  net, net_ctc = nets # net contains text-localization FCN, net_ctc contains text-recognition FCN
  img = batch[0]
  
  #imgo = batch[4][0]
  
  baseName = os.path.basename(batch[3][0]) 
  inputDir = os.path.dirname(batch[3][0])
  dict_file  = '{0}/per_image/voc_{1}.txt'.format(inputDir, baseName[:-4])
  if os.path.exists(dict_file) and use_per_image:
      cmp_trie.load_dict(dict_file)
  
  #img = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
  #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  
  im = np.asarray(img, dtype=np.float)
  im = im / 128.0
  im = im - 1.0
  #im = im.reshape((3, im.shape[0], im.shape[1]))
  im = np.swapaxes(im,1,3)
  im = np.swapaxes(im,2,3)
  net.blobs['data'].reshape(im.shape[0],im.shape[1],im.shape[2],im.shape[3]) # Reshape network's data blob to adapt input image size
  net.blobs['data'].data[...] = im # Load image onto network's data blob
  net.reshape()
  start = time.time()
  out = net.forward(start='conv1') # Feed-forward for localizetion FCN
  end = time.time()
  seconds = end - start
  fps = 1 / seconds # Compute frames per second localization FCN is able to process
  #print("loc fps:{0}".format(fps))

  boxes  = out['boxes'] # Generated boxes from localization FCN's 'boxes' blob
  if out.has_key('boxes0'):
    boxes0  =  net.blobs['boxes0'].data[...]
    boxes = np.concatenate((boxes, boxes0), axis = 2)
    boxes = boxes[:, :, np.argsort(boxes[0, 0, :, 5])[::-1], :]
  
  detections_out = ocr_detections(net_ctc, batch[4][0], batch[0][0], boxes, image_size, r_p_th, out_raw, baseName, debug, split_words, alow_non_dict)
      
  return detections_out
    
    
def validate(nets, dataloader, image_size = [480, 480], split_words = True):
    
  cmp_trie.load_dict('/home/busta/data/icdar2013-Test/GenericVocabulary.txt')
  
  net0, net_ctc0 = nets
  net = net0.test_nets[0]
  net.share_with(net0.net)
  net_ctc = net_ctc0.test_nets[0]
  net_ctc.share_with(net_ctc0.net)
  
  tp_all = 0
  gt_all = 0
  tp_e2e_all = 0
  gt_e2e_all = 0
  dataloader.reset()
  cnt = 0
  while dataloader.has_next('valid'):
    batch, image_size = dataloader.get_next('valid', 1, image_size = image_size)
    detections_out = forward_image([net,net_ctc], batch, image_size, r_p_th=0.05, split_words=split_words)
    
    word_gt_orig = batch[5][0]
    tp, tp_e2e, gt_e2e = evaluate_image(batch, detections_out, word_gt_orig)
    
    word_gto = batch[1][0]
    tp_all += tp 
    gt_all += len(word_gto)
    tp_e2e_all += tp_e2e
    gt_e2e_all += gt_e2e
    print("  E2E recall {0:.3f} / {1:.3f}".format( tp_e2e_all / float( max(1, gt_e2e_all) ), tp / float( max(1, gt_e2e )) ))
    if cnt > 20:
      break
    cnt += 1
  print("E2E recall {0:.3f} / {1:.3f}".format( tp_e2e_all / float( max(gt_e2e_all, 1) ), tp_all / float( max(1, gt_e2e_all) ) ))
    
        
