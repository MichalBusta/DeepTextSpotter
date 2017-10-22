# -*- coding: utf-8 -*-
'''
Created on Mar 1, 2017

@author: Michal.Busta at gmail.com
'''
import numpy as np
import math
import vis

import cv2
import cmp_trie

def intersect(a, b):
  '''Determine the intersection of two rectangles'''
  rect = (0,0,0,0)
  r0 = max(a[0],b[0])
  c0 = max(a[1],b[1])
  r1 = min(a[2],b[2])
  c1 = min(a[3],b[3])
  # Do we have a valid intersection?
  if r1 > r0 and  c1 > c0: 
      rect = (r0,c0,r1,c1)
  return rect

def union(a, b):
  r0 = min(a[0],b[0])
  c0 = min(a[1],b[1])
  r1 = max(a[2],b[2])
  c1 = max(a[3],b[3])
  return (r0,c0,r1,c1)

def area(a):
  '''Computes rectangle area'''
  width = a[2] - a[0]
  height = a[3] - a[1]
  return width * height


codec = u' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~£ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž'
#codec = u' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~£ACDEEINORSTUUYZacdeeinorstuuyz'


def print_seq(wf):
  prev = 0
  word = ''
  conf = 0
  for cx in range(0, wf.shape[0]):
    c = wf[cx, 0]
    if prev == c:
      continue
    if c > 3 and c < 125:
      ordv = codec[c - 4]
      char = ordv
      word += char
    
    prev = c    
  return word.strip()

def print_seq_ext(wf, conf):
  prev = 0
  word = ''
  start_pos = 0
  end_pos = 0
  dec_splits = []
  hasLetter = False
  for cx in range(0, wf.shape[0]):
    c = wf[cx]
    if prev == c:
      if c > 2:
          end_pos = cx
      continue
    if c > 3 and c < 140:
      ordv = codec[c - 4]
      char = ordv
      if char == ' ':
        if hasLetter:
          dec_splits.append(cx + 1)
      else:
        hasLetter = True
      word += char
      end_pos = cx
    elif c > 0:
      if hasLetter:
        dec_splits.append(cx + 1)
        word += ' '
        end_pos = cx
      
    
    if len(word) == 0:
      start_pos = cx
    
    prev = c    
  conf2 = [conf, start_pos, end_pos + 1]
  return word.strip(), np.array([conf2]), np.array([dec_splits])

def print_seq2(wf):
  prev = 0
  word = ''
  for cx in range(0, wf.shape[0]):
    c = wf[cx]
    if c > 0 and c < 128:
      char = unichr(c)
      word += char
    prev = c    
  return word.strip()

def box_to_affine(xc, yc, angle, scalex, aspect):
    
  m = np.zeros((2, 3), np.double )
  
  m[0,0] = scalex * math.cos(angle)
  m[1,0] = scalex * math.sin(angle) 
  m[0,1] = - scalex * math.sin(angle) * aspect
  m[1,1] = scalex * math.cos(angle) * aspect
  m[0,2] = xc 
  m[1,2] = yc 
  
  return m

def get_normalized_image(img, rr, debug = False):

  box = cv2.boxPoints(rr)
  extbox = cv2.boundingRect(box)

  if extbox[2] *  extbox[3] > img.shape[0] * img.shape[1]:
    print("Too big proposal: {0}x{1}".format(extbox[2], extbox[3]))
    return None, None
  extbox = [extbox[0], extbox[1], extbox[2], extbox[3]]
  extbox[2] += extbox[0]
  extbox[3] += extbox[1]
  extbox = np.array(extbox, np.int)
  
  extbox[0] = max(0, extbox[0])
  extbox[1] = max(0, extbox[1])
  extbox[2] = min(img.shape[1], extbox[2])
  extbox[3] = min(img.shape[0], extbox[3])
  
  tmp = img[extbox[1]:extbox[3], extbox[0]:extbox[2]]
  center = (tmp.shape[1] / 2,  tmp.shape[0] / 2)
  rot_mat = cv2.getRotationMatrix2D( center, rr[2], 1 )
  
  if tmp.shape[0] == 0 or tmp.shape[1] == 0:
    return None, rot_mat
  
  if debug:
    vis.draw_box_points(img,  np.array(extbox, dtype="int"), color = (0, 255, 0))
    cv2.imshow('scaled', img)

  rot_mat[0,2] += rr[1][0] /2.0 - center[0]
  rot_mat[1,2] += rr[1][1] /2.0 - center[1]
  try:
    norm_line = cv2.warpAffine( tmp, rot_mat, (int(rr[1][0]), int(rr[1][1])), borderMode=cv2.BORDER_REPLICATE )
  except:
    return None, rot_mat
  return norm_line, rot_mat

def get_obox(im, scaled, box):
        
  image_size = (scaled.shape[1], scaled.shape[0])
  
  o_size = (im.shape[1], im.shape[0])
  scalex = o_size[0] / float(image_size[0])
  scaley = o_size[1] / float(image_size[1])
  
  box2 = np.copy(box)

  gtbox  = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), box[2])
  gtbox = cv2.boxPoints(gtbox)
  gtbox = np.array(gtbox, dtype="float")
  
  #vis.draw_box_points(im,  np.array(gtbox, dtype="int"), color = (0, 255, 0))
  #cv2.imshow('orig', im)
  
  gtbox[:,0] /=  scalex
  gtbox[:,1] /=  scaley
  
  dh = gtbox[0, :] - gtbox[1, :]
  dw = gtbox[1, :] - gtbox[2, :]
  
  
  h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1])
  w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
  
  box2[0][0] /=  scalex
  box2[0][1] /=  scaley
  
  box2[1][0] = w
  box2[1][1] = h
  
  box2[2] = math.atan2((gtbox[2, 1] - gtbox[1, 1]), (gtbox[2, 0] - gtbox[1, 0])) * 180 / 3.14
      
  return box2  


def process_splits(trans, conf, splits, norm2, ctc_f, rot_mat, boxt, draw, iou, debug = False, alow_non_dict = False):

  '''
  Summary : Split the transciption and corresponding bounding-box based on spaces predicted by recognizer FCN.
  Description : 

  Parameters
  ----------
  trans : string
      String containing the predicted transcription for the corresponding predicted bounding-box.
  conf : list
      List containing sum of confidence for all the character by recognizer FCN, start and end position in bounding-box for generated transciption.
  splits :  list
      List containing index of position of predicted spaces by the recognizer FCN.
  norm2 : matrix
      Matrix containing the cropped bounding-box predicted by localization FCN in the originial image.
  ctc_f : matrix
      Matrix containing output of recognizer FCN for the given input bounding-box.
  rot_mat : matrix
      Rotation matrix returned by get_normalized_image function.
  boxt : tuple of tuples
      Tuple of tuples containing parametes of predicted bounding-box by localization FCN.
  draw : matrix
      Matrix containing input image. 
  debug : boolean
      Boolean parameter representing debug mode, if it is True visualization boxes are generated.
          
  Returns
  -------
  boxes_out : list of tuples
      List of tuples containing predicted bounding-box parameters, predicted transcription and mean confidence score from the recognizer.
  '''
  
  spl = trans.split(" ")
  boxout = cv2.boxPoints(boxt)
  start_f = 0
  mean_conf = conf[0, 0] / len(trans) # Overall confidence of recognizer FCN
  boxes_out = []
  
  for s in range(len(spl)):
      
    text = spl[s]
    
    end_f = conf[0, 2]
    if s < len(spl) - 1:
      try:
        if splits[0, s] > start_f:
          end_f = splits[0, s] # New ending point of bounding-box transcription
      except IndexError:
        pass
    
    scalex = norm2.shape[1] / float(ctc_f.shape[0])
        
    poss = start_f * scalex
    pose = (end_f + 2) * scalex
    rect = [[poss, 0], [pose, 0], \
            [pose, norm2.shape[0] - 1], [poss, norm2.shape[0] - 1]]
    rect = np.array(rect)
    #rect[:, 0] +=  boxt[0][0]
    #rect[:, 1] += boxt[0][1]
    
    int_t = cv2.invertAffineTransform(rot_mat)
    
    dst_rect = np.copy(rect)
    dst_rect[:,0]  = int_t[0,0]*rect[:,0] + int_t[0,1]*rect[:, 1] + int_t[0,2]
    dst_rect[:,1]  = int_t[1,0]*rect[:,0] + int_t[1,1]*rect[:, 1] + int_t[1,2]
    
    
    tx = np.sum(dst_rect[:,0]) / 4.0
    ty = np.sum(dst_rect[:,1]) / 4.0
    br = cv2.boundingRect(boxout)
    tx += br[0]
    ty += br[1]
    twidth = (pose - poss) #twidth = (pose - poss) / ext_factor
    theight = norm2.shape[0]
    
    
    box_back = ( (tx, ty), (twidth, theight * 0.9), boxt[2] )
    
    if debug:
      boxout_u = cv2.boxPoints(box_back)
      vis.draw_box_points(draw, boxout_u, color = (0, 255, 0))
      cv2.imshow('draw', draw)
        
    if len(text.strip()) == 0:
      print("zero length text!")
      continue 
    
    textc = text.replace(".", "").replace(":", "").replace("!", "").replace("?", "").replace(",", "").replace("/", "").replace("-", "").replace("$", "").replace("'", "").replace("(", "").replace(")", "").replace("+", "")
    if textc.endswith("'s"):
      textc = textc[:-2]
    is_dict = cmp_trie.is_dict(textc.encode('utf-8')) or textc.isdigit() or alow_non_dict
    if len(text) > 2 and ( text.isdigit() or is_dict):
        boxes_out.append( (box_back, (text, mean_conf, is_dict, iou) ) )
    start_f = end_f + 1      
  return boxes_out    
