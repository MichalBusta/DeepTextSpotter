# -*- coding: utf-8 -*-
import numpy as np
import sys, os
sys.path.insert(0, '/usr/local/python/')
baseDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{0}/build'.format(baseDir))

import caffe

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

import cv2

import math

import random 
from models import create_solvers_tiny

from data import DataLoader

import vis
import matplotlib.pyplot as plt

import argparse

from validation import validate    
image_no = 0

import utils
from utils import intersect, union, area, get_normalized_image, get_obox


buckets = [54, 80, 124, 182, 272, 410, 614, 922, 1383, 2212]  
image_sizes = [[352, 352], [416, 416] ] #,[480, 480], [544, 544], [576, 576]]    
image_size = [160, 160]
it = 0
mean_loss = 0
mean_rec = 0

valid_interval = 100
snapshot_interval = 1000

codec = u' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~£ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž'
codec_rev = {}
index = 4
for i in range(0, len(codec)):
  codec_rev[ord(codec[i])] = index
  index += 1
    

def process_batch(nets, optim, optim2, image_size, args):
  global it, mean_loss, mean_rec
  
  net, net_ctc = nets
  
  net = net.net
  net_ctc = net_ctc.net
  
  
  net.blobs['data'].reshape(args.batch_size,1,image_size[1],image_size[0])
  net.reshape()
      
  it += 1 
  
  optim2.step(1)
  
  im = net.blobs['data'].data[...]
  draw = np.swapaxes(im,2,3)
  draw = np.swapaxes(draw,1,3)
  im_ctc = np.copy(draw)
  draw += 1
  draw *= 128
  draw = np.array(draw, dtype="uint8").copy() 
  
  
  if args.debug:
    grid_step = 16
    line = 0
    while line < image_size[0]:
      cv2.line(draw[0], (0, line), (image_size[1], line), (128, 128, 128))
      line += grid_step
  
  boxes  =  net.blobs['boxes'].data[...]
                 
  word_gtob = net.blobs['gt_boxes'].data[...]
  word_txt = net.blobs['gt_labels'].data[...]
  
  lines_gtob = net.blobs['line_boxes'].data[...]
  lines_txt = net.blobs['line_labels'].data[...]
  
  #nms = boxeso[:, 0, 0, 8] == 0
  #boxes = boxes[:, :, nms, :]
  
  boxes[:, 0, :, 0] *= image_size[0]
  boxes[:, 0, :, 1] *= image_size[1]
  normFactor = math.sqrt(image_size[1] * image_size[1] + image_size[0] * image_size[0])
  boxes[:, 0, :, 2] *= normFactor
  boxes[:, 0, :, 3] *= normFactor
  
  sum_cost = 0
  count = 0
  
  labels_gt = []
  labels_det = []
  
  gt_to_detection = {}
  net_ctc.clear_param_diffs()
  
  
  batch_buckets = []    
  dummy = {} 
  
  matched_detections = 0
  for bid in range(im.shape[0]):
    
    o_image = net.layers[0].get_image_file_name(bid)
    o_image = cv2.imread(o_image, cv2.IMREAD_GRAYSCALE)
    cx = net.layers[0].get_crop(bid, 0)
    cy = net.layers[0].get_crop(bid, 1)
    cmx = net.layers[0].get_crop(bid, 2)
    cmy = net.layers[0].get_crop(bid, 3)
    o_image = o_image[cy:cmy, cx:cmx]
    
    boxes_count = 0
    for i in range(0, boxes.shape[2]):
      det_word = boxes[bid, 0, i]
      if (det_word[0] == 0 and det_word[1] == 0) or det_word[5] < 0.01:
          break
      boxes_count += 1
        
    x = [i for i in range(boxes_count)]
    #random.shuffle(x)
    
    bucket_images = {}
    batch_buckets.append(bucket_images)
    
    word_gto = word_gtob[bid]
    word_gto_txt = word_txt[bid]
    gt_count = 0 
    for gt_no in range(word_gto.shape[0]):
      gt = word_gto[gt_no, :]
      gt = gt.reshape(6)
      gtnum = 1000 * bid +  gt_no
      
      if gt[5] == -1:
        #print("ignore gt!")
        continue
      
      gt_count += 1
                  
      txt = word_gto_txt[gt_no, :]
      gtbox  = ((gt[0] * image_size[0], gt[1] * image_size[1]), (gt[2] * normFactor, gt[3] * normFactor), gt[4] * 180 / 3.14)
      gtbox = cv2.boxPoints(gtbox)
      
      gtbox = np.array(gtbox, dtype="int")
      rect_gt = cv2.boundingRect(gtbox)

      if rect_gt[0] == 0 or rect_gt[1] == 0 or  rect_gt[0] + rect_gt[2]  >= image_size[0] or rect_gt[1] + rect_gt[3]  >= image_size[1]:
        continue
      
      if gt[3] * normFactor <  3:
        if args.debug:
          #print('too small gt!')
          vis.draw_box_points(draw[bid], gtbox, color = (255, 255, 0))
          cv2.imshow('draw', draw[bid])
        continue
        
      if args.debug:
        vis.draw_box_points(draw[bid], gtbox, color = (0, 0, 0), thickness=2)
      
      #vis.draw_box_points(draw[bid], gtbox, color = (255, 255, 255))
      #cv2.imshow('draw', draw[bid])
      
      rect_gt = [rect_gt[0], rect_gt[1], rect_gt[2], rect_gt[3]]
      rect_gt[2] += rect_gt[0]
      rect_gt[3] += rect_gt[1]

      for i in range(0, min(100, boxes_count)):
        if math.fabs(gt[4] - det_word[4]) > math.pi / 16:
          continue
        
        det_word = boxes[bid, 0, x[i], :]
        
        if (det_word[0] == 0 and det_word[1] == 0) or det_word[5] < 0.01:
          break
        
        box  = ((det_word[0], det_word[1]), (det_word[2], det_word[3]), det_word[4] * 180 / 3.14)
        box = cv2.boxPoints(box)
        
        if args.debug:
          boxp = np.array(box, dtype="int")
          vis.draw_box_points(draw[bid], boxp, color = (0, 255, 0))
        
        box = np.array(box, dtype="int")
        bbox = cv2.boundingRect(box)
        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
   
        #rectangle intersection ... 
        inter = intersect(bbox, rect_gt)
        uni = union(bbox, rect_gt)
        ratio = area(inter) / float(area(uni))
        
        ratio_gt = area(inter) / float(area(rect_gt))
        if ratio_gt < 0.95:
          continue 
        
        if ratio < 0.5:
          continue
        
        if not gt_to_detection.has_key(gtnum):
            gt_to_detection[gtnum] = [0, 0, 0]
        tupl = gt_to_detection[gtnum] 
        if tupl[0] < ratio:
          tupl[0] = ratio 
          tupl[1] = x[i]  
          tupl[2] = ratio_gt       
        
        det_word = boxes[bid, 0, x[i], :]
        box  = ([det_word[0], det_word[1]], [det_word[2], det_word[3]], det_word[4] * 180 / 3.14)
        
        boxO = get_obox(im_ctc[bid], o_image, box)
        boxO = ((boxO[0][0], boxO[0][1]), (boxO[1][0], boxO[1][1]), boxO[2])
        norm2, rot_mat = get_normalized_image(o_image, boxO)
        #norm3, rot_mat = get_normalized_image(im_ctc[bid], ([det_word[0], det_word[1]], [det_word[2] * 1.2, det_word[3] * 1.1], det_word[4] * 180 / 3.14))
        if norm2 is None:
          continue
        #if norm3 is None:
        #  continue
        #continue
        #cv2.imshow('ts', norm2)
        #cv2.imshow('ts3', norm3)
        #cv2.waitKey(1)
        width_scale = 32.0 / norm2.shape[0]
        width = norm2.shape[1] * width_scale
        best_diff = width
        bestb = 0
        for b in range(0, len(buckets)):
          if best_diff > abs(width * 1.3 - buckets[b]):
            best_diff = abs(width * 1.3 - buckets[b])
            bestb = b
        
        scaled = cv2.resize(norm2, (buckets[bestb], 32))  
        scaled = np.asarray(scaled, dtype=np.float)
        delta = scaled.max() - scaled.min()
        scaled = (scaled) / (delta / 2)
        scaled -= scaled.mean()
                
        if not bucket_images.has_key(bestb):
          bucket_images[bestb] = {}
          bucket_images[bestb]['img'] = []  
          bucket_images[bestb]['sizes'] = []    
          bucket_images[bestb]['txt'] = []
          bucket_images[bestb]['gt_enc'] = []
          dummy[bestb] = 1
        else:
          if args.debug and len(bucket_images[bestb]) > 4:
            continue    
          elif  len(bucket_images[bestb]) > 32:
            continue
        
        gt_labels = []
        txt_enc = ''
        for k in range(txt.shape[1]):
          if txt[0, k] > 0:
            if codec_rev.has_key(txt[0, k]):                
              gt_labels.append( codec_rev[txt[0, k]] )
            else:
              gt_labels.append( 3 )
                              
            txt_enc += unichr(txt[0, k])
          else:
            gt_labels.append( 0 )
        
        if scaled.ndim == 3:
          scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        if args.debug:
          cv2.imshow('scaled', scaled)
        bucket_images[bestb]['sizes'].append(len(gt_labels))
        bucket_images[bestb]['gt_enc'].append(gt_labels)
        bucket_images[bestb]['txt'].append(txt_enc)
        bucket_images[bestb]['img'].append(scaled)
        matched_detections += 1   
      
  #and learn OCR
  for bucket in bucket_images.keys():
      
    imtf = np.asarray(bucket_images[bucket]['img'], dtype=np.float)
    imtf = np.reshape(imtf, (imtf.shape[0], -1, imtf.shape[1], imtf.shape[2]))    
    #imtf = imtf.reshape((imtf.shape[0], imtf.shape[1], imtf.shape[2], 1))
    #imtf = np.swapaxes(imtf,1,3)
    
    
    net_ctc.blobs['data'].reshape(imtf.shape[0],imtf.shape[1],imtf.shape[2], imtf.shape[3]) 
    net_ctc.blobs['data'].data[...] = imtf
    
    labels = bucket_images[bucket]['gt_enc']
    txt = bucket_images[bucket]['txt']
    
    max_len = 0
    for l in range(0, len(labels)):
      max_len = max(max_len, len(labels[l]))
    for l in range(0, len(labels)):
      while len(labels[l]) <  max_len:
        labels[l].append(0)
      
    
    labels = np.asarray(labels, np.float)
    
    net_ctc.blobs['label'].reshape(labels.shape[0], labels.shape[1])
    
    net_ctc.blobs['label'].data[...] = labels
    
    if args.debug:
        vis.vis_square(imtf[0])
        cv2.imshow('draw', draw[0])
        cv2.waitKey(5)
         
     
    optim.step(1)  
    sum_cost += net_ctc.blobs['loss'].data[...]
    if net_ctc.blobs['loss'].data[...] > 10:
      vis.vis_square(imtf[0])
      cv2.imshow('draw', draw[0])
      sf = net_ctc.blobs['transpose'].data[...]
      labels2 = sf.argmax(3)
      out = utils.print_seq(labels2[:, 0, :])
      print(u'{0} - {1}'.formayolo_mobile_iter_0t(out, txt[0])  )
      cv2.waitKey(5)
          
          
    count += imtf.shape[0]
              
  correct_cout = 0    
  for i in range(len(labels_gt)):
    det_text = labels_det[i]
    gt_text = labels_gt[i]
    
    if it % 100 == 0:
      print( u"{0} - {1}".format(det_text, gt_text).encode('utf8') )
    if det_text == gt_text:
      correct_cout += 1
      
  count = max(count, 1)    
  mean_loss = 0.99 * mean_loss + 0.01 * sum_cost / count
  mean_rec = mean_rec * 0.99 + 0.01 * correct_cout / float(max(1, len(labels_gt)))
  
  #count detection ratio

  tp = 0
  for bid in range(im.shape[0]):
    word_gto = word_gtob[bid]
    for gt_no in range(len(word_gto)):
      gt = word_gto[gt_no]
      gtnum = 1000 * bid +  gt_no
      if gt_to_detection.has_key(gtnum):
        tupl = gt_to_detection[gtnum] 
        if tupl[0] > 0.5:
          tp += 1
          
                      
  loc_recall = tp / float(max(1, gt_count))             
  if args.debug:
    cv2.imshow('draw', draw[0])
    if im.shape[0] > 1:
        cv2.imshow('draw2', draw[1])
        
    cv2.waitKey(10)
  
  if it % 10 == 0:
    print('{0} - lr:{1:.3e} ctc:{2:.4f}/{3:.4f} wr:{4:.2f}/{5:.2f}, loc:{6:.2f} {7}'.format(it, 0.0001, sum_cost / count, mean_loss, correct_cout / float(max(1, len(labels_gt))), mean_rec, loc_recall, matched_detections))
  
  if it % 1000 == 0:
    optim.snapshot()
    optim2.snapshot()
  
    
        
def train_dir(nets, optim, optim2, dataloader, args):
    
  global image_size, it, image_sizes
  caffe.set_mode_gpu() 
  
  if args.debug:
    image_sizes = [[416, 416]]

  while True:
      
    if it % 500 == 0:
      image_size = image_sizes[random.randint(0, len(image_sizes) - 1)]
      print(image_size)
    
    #im = cv2.imread('/home/busta/data/90kDICT32px/background/n03085781_3427.jpg')
    #try:
    process_batch(nets, optim, optim2, image_size, args)
    
    if it % valid_interval == 0:
      validate(nets, dataloader, image_size = [416, 416], split_words=False)
        
    #except:
    #    continue
        
        
parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default='/mnt/textspotter/tmp/SynthText')
parser.add_argument('-train_list', default='/home/busta/data/test_icdar181.txt')
parser.add_argument('-valid_list', default='/home/busta/data/ocr/list.txt')
parser.add_argument('-debug', type=int, default=0)
parser.add_argument('-batch_size', type=int, default=4)

args = parser.parse_args()

        
nets = create_solvers_tiny(args)
yolonet = nets[0]        
net_ctc = nets[1]    
sgd = net_ctc
dataloader = DataLoader({'train': args.train_list, 'valid': args.valid_list}, \
                         root_dir = args.data_dir, gray_scale=True)

train_dir(nets, sgd, yolonet, dataloader, args)



