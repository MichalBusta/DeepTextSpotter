'''
Created on Mar 1, 2017

@author: busta
'''

import os
import random

import codecs
import unicodecsv

import numpy as np

import cv2
import math

from vis import draw_box_points

def read_txt_gt(gt_file, multi_script = False, separator = ' ', skip_slash = False):
     
  f = codecs.open(gt_file, "rb", "utf-8-sig")
  lines = f.readlines()
  
  gt_rectangles = []
  for line in lines:
    if line[0] == '#':
      continue
    splitLine = line.split(separator);
    if len(splitLine) < 7:
      continue
    xline = u'{0}'.format(line.strip())
    xline = xline.encode('utf-8') 
    
    for splitLine in unicodecsv.reader([xline], skipinitialspace=True, quotechar='"', delimiter=separator, encoding='utf-8'):
      break
    
    language = ""
    cls = splitLine[0].strip()
    x =  float(splitLine[1].strip())
    y = float(splitLine[2].strip())
    w = float(splitLine[3].strip())
    h = float(splitLine[4].strip())
    angle = float(splitLine[5].strip())
    if multi_script == True:
      text = splitLine[7].strip()
      language = splitLine[6].strip()
    else:
      delim = ""
      text = ""
      for i in range(6, len(splitLine)):
        text += delim + splitLine[i].strip()
        delim = " "
        
    if text[0] == '#' and skip_slash:
      continue

    gt_rectangles.append( [ x, y, w, h, angle, text, cls, 0, 0, -1, -1, "", -1, language ] )
          
  if len(gt_rectangles) == 0:
    raise ValueError()
  return gt_rectangles      

class DataLoader(object):
  '''
  classdocs
  '''
  def __init__(self, sets, root_dir = '/', gray_scale = False, load_gt = True, debug=False):
      '''
      Constructor
      '''
      
      self.data = {}
      for key in sets.keys():
          list_file = sets[key]
          images = []
          with open(list_file, "r") as ins:
              for line in ins:
                  images.append(line.strip())
                  
          self.data[key] = images
          self.counter = 0
          self.it = 0
          self.step = 50
          self.crop_ratio = 0.3
          
      self.root_dir = root_dir
      self.gray_scale = gray_scale
      self.load_gt = load_gt
      self.debug = debug
          
  def random_crop(self, img, word_gto):
      
      xs =  int(random.uniform(0, self.crop_ratio) * img.shape[1])
      xe =  int(random.uniform(0, self.crop_ratio) * img.shape[1])
      maxx = img.shape[1] - xe
      
      ys =  int(random.uniform(0, self.crop_ratio) * img.shape[0])
      ye =  int(random.uniform(0, self.crop_ratio) * img.shape[0])
      maxy = img.shape[0] - ye
      
      crop_img = img[ys:maxy, xs:maxx]
      
      normo = math.sqrt(img.shape[0] * img.shape[0] + img.shape[1] * img.shape[1] )
      
      image_size = (crop_img.shape[1], crop_img.shape[0]) 
      normo2 = math.sqrt(image_size[1] * image_size[1] + image_size[0] * image_size[0] )
      
      o_size = (img.shape[1], img.shape[0])
      
      gt_out = []
      for gt_no in range(len(word_gto)): #TODO - remove loop ... use numpy
          
          gt = word_gto[gt_no]
          
          gtbox  = ((gt[0] * o_size[0], gt[1] * o_size[1]), (gt[2] * normo, gt[3] * normo), gt[4] * 180 / 3.14)
          gtbox = cv2.boxPoints(gtbox)
          gtbox = np.array(gtbox, dtype="float")
          
          gtbox[:, 0] -= xs
          gtbox[:, 1] -= ys
          
          gtbox[gtbox[:, 0] < 0, 0] = 0
          gtbox[gtbox[:, 1] < 0, 1] = 0
          
          gtbox[gtbox[:, 0] > maxx, 0] = maxx
          gtbox[gtbox[:, 1] > maxy, 1] = maxy
          
          dh = gtbox[0, :] - gtbox[1, :]
          dw = gtbox[1, :] - gtbox[2, :]
          
          centerx = np.sum( gtbox[:, 0] ) / float(gtbox.shape[0])
          centery = np.sum( gtbox[:, 1] ) / float(gtbox.shape[0])
          
          
                      
          h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) / normo2
          w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1]) / normo2
          
          if w * normo2 < 2 or h * normo2  < 2 or np.isinf(w) or np.isinf(h):
              #print("warn: removig too small gt {0}".format(gt))
              continue
          
          gt[0] = centerx / image_size[0]
          gt[1] = centery / image_size[1]
          gt[2] = w
          gt[3] = h
          
          gt[4] = math.atan2((gtbox[2, 1] - gtbox[1, 1]), (gtbox[2, 0] - gtbox[1, 0]))
          
          if False:
              draw_box_points(crop_img,  np.array(gtbox, dtype="int"), color = (0, 255, 0))
              
              gtbox2  = ((gt[0] * image_size[0], gt[1] * image_size[1]), (gt[2] * normo2, gt[3] * normo2), gt[4] * 180 / 3.14)
              gtbox2 = cv2.boxPoints(gtbox2)
              gtbox2 = np.array(gtbox2, dtype="float")
              
              draw_box_points(crop_img,  np.array(gtbox2, dtype="int"), color = (0, 255, 0))
              cv2.imshow('c2', crop_img)
              
          gt_out.append(gt)
          
      #cv2.imshow('crop_img', crop_img)
      #cv2.waitKey(0)
      return (crop_img, gt_out)
  
  
  def transform_boxes(self, im, scaled, word_gto):
      
      image_size = (scaled.shape[1], scaled.shape[0])
      
      o_size = (im.shape[1], im.shape[0])
      normo = math.sqrt(im.shape[1] * im.shape[1] + im.shape[0] * im.shape[0] )
      normo2 = math.sqrt(image_size[1] * image_size[1] + image_size[0] * image_size[0] )
      scalex = o_size[0] / float(image_size[0])
      scaley = o_size[1] / float(image_size[1])
      
      gto_out = []
      for gt_no in range(len(word_gto)):
          gt = word_gto[gt_no]
          gtbox  = ((gt[0] * o_size[0], gt[1] * o_size[1]), (gt[2] * normo, gt[3] * normo), gt[4] * 180 / 3.14)
          gtbox = cv2.boxPoints(gtbox)
          gtbox = np.array(gtbox, dtype="float")
          
          #draw_box_points(im,  np.array(gtbox, dtype="int"), color = (0, 255, 0))
          #cv2.imshow('orig', im)
          
          gtbox[:,0] /=  scalex
          gtbox[:,1] /=  scaley
          
          dh = gtbox[0, :] - gtbox[1, :]
          dw = gtbox[1, :] - gtbox[2, :]
          
          
          h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) / normo2
          w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1]) / normo2
          
          #if w * normo2 < 5 or h * normo2  < 5 or np.isinf(w) or np.isinf(h):
              #print("warn: removig too small gt {0}".format(gt))
          #    continue
          
          gt[2] = w
          gt[3] = h
          
          gt[4] = math.atan2((gtbox[2, 1] - gtbox[1, 1]), (gtbox[2, 0] - gtbox[1, 0]))
          
          gt[8] = gtbox
          if self.debug:
            print('gtbox : ' + str(gtbox))
          gto_out.append(gt)
          #draw_box_points(scaled,  np.array(gtbox, dtype="int"), color = (0, 255, 0))
          #cv2.imshow('scaled', scaled)
          #cv2.waitKey(0)
          
      return gto_out
  
      
  def get_batch(self, set_name, count, image_size = [544, 544]):
      
      images_read = 0
      image_list = self.data[set_name]
      images = []
      gt_ret = []
      scales = []
      names = []
      imageso = []
      if self.it % 30 == 0:
          self.step += 1
      self.it += 1    
      self.step = min(self.step, len(image_list) - 1) 
      while True:    
          imageNo = random.randint(0, len(image_list) - 1)
          image_name = image_list[imageNo]
          
          if image_name[0] != '/':
              image_name = '{0}/{1}'.format(self.root_dir, image_name)
          lineGt = image_name.replace(".jpg", ".txt")
          if os.path.exists(lineGt):
              try:
                  word_gto = read_txt_gt(lineGt, skip_slash = imageNo < 10000)
              except:
                  print("invalid gt: {0}".format(lineGt))
                  continue
          else:
              print("no gt: {0}".format(lineGt))
              continue
          
          im = cv2.imread(image_name)
          #(im, word_gto) = self.random_crop(im, word_gto)
          
          scaled = cv2.resize(im, (image_size[0], image_size[1]))
          if self.gray_scale:
              scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY )
              scaled = scaled.reshape((scaled.shape[0], scaled.shape[1], 1))
          
          gto_out = self.transform_boxes(im, scaled, word_gto)
          
          names.append(image_name)
          scales.append((image_size[0] / float(scaled.shape[1]), image_size[1] / float(scaled.shape[0])  ))
          
          images.append(scaled)
          gt_ret.append(gto_out)
          imageso.append(im)
          images_read += 1
          
          if images_read >= count:
              break
          
      return (images, gt_ret, scales, names, imageso)
    
  def get_best_image_size(self, img):
    
    scale = 1.5
    image_sizes = [[352, 352], [416, 416], [480, 480], [544, 544], [576, 576], [608, 608]]#, [640, 640] ]    
    best_width = image_sizes[0][0]
    best_height = image_sizes[0][0]
    for ims in  image_sizes:
      if abs(best_width - img.shape[1] / scale) > abs(ims[0] - img.shape[1] / scale):
        best_width = ims[0]
        
      if abs(best_height - img.shape[0] / scale) > abs(ims[0] - img.shape[0] / scale):
        best_height = ims[0]
    
    print([best_width, best_height])    
    return [best_width, best_height]                
    
  
  def get_next(self, set_name, count, multi_script = False, image_size = [544, 544]):
      
      images_read = 0
      image_list = self.data[set_name]
      images = []
      gt = []
      gt_orig = []
      scales = []
      imageso = []
      names = []
      
      while True:    
          
          imageNo = self.counter
          self.counter += 1
          
          image_name = image_list[imageNo]
          if image_name[0] != '/':
              image_name = '{0}/{1}'.format(self.root_dir, image_name)
          
          
          #image_name= '/home/busta/data/icdar2013-Test/img_126.jpg'
          #image_name= '/home/busta/data/icdar2015-Ch4-Train/img_108.jpg'
          #image_name = '/home/busta/data/icdar2013-Test/img_12.jpg'    
          #image_name = '/home/busta/data/icdar2013-Test/img_106.jpg'
          lineGt = image_name.replace(".jpg", ".txt")
          lineGt = lineGt.replace(".png", ".txt")
          if os.path.exists(lineGt):
              try:
                  word_gto = read_txt_gt(lineGt, multi_script, skip_slash = imageNo < 10000)
              except:
                  print("invalid gt: {0}".format(lineGt))
                  if self.load_gt:
                    word_gto = []
                    #continue
                  else:
                      word_gto = []
                      
          else:
              print("no gt: {0}".format(lineGt))
              word_gto = []
              #continue
              
          if self.debug:
            print(str(image_name)) 
          im = cv2.imread(image_name)
          #image_size = self.get_best_image_size(im)
          #image_size = [im.shape[1]  /  32 * 32, im.shape[0] /  32 * 32] 
          #while image_size[0] > 1024 or image_size[1] > 1024:
          #    image_size[0] /= 1.2
          #    image_size[1] /= 1.2   
          #    image_size[0] = int(image_size[0]) /32 * 32
          #    image_size[1] = int(image_size[1]) /32 * 32
                
          scaled = cv2.resize(im, (image_size[0], image_size[1]))
          
          if self.gray_scale:
            scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY )
            scaled = scaled.reshape((scaled.shape[0], scaled.shape[1], 1))
        
          if self.debug:
            print 'word_gto : ' + str(word_gto)
          gto_out = self.transform_boxes(im, scaled, word_gto)
          scales.append((image_size[0] / float(scaled.shape[1]), image_size[1] / float(scaled.shape[0])  ))
          
          images.append(scaled)
          gt.append(gto_out)
          gt_orig.append(word_gto)
          imageso.append(im)
          
          images_read += 1
          names.append(image_name)
          
          if images_read >= count:
              break
          
      return (images, gt, scales, names, imageso, gt_orig), image_size
  
  
  def get_same(self, set_name, count, multi_script = False, image_size = [544, 544]):
      
      images_read = 0
      image_list = self.data[set_name]
      images = []
      gt = []
      gt_orig = []
      scales = []
      imageso = []
      names = []
      
      while True:    
          
          imageNo = self.counter - 1
          
          image_name = image_list[imageNo]
          if image_name[0] != '/':
              image_name = '{0}/{1}'.format(self.root_dir, image_name)
          
          #image_name = '/home/busta/data/icdar2013-Test/img_12.jpg'  
          #image_name = '/home/busta/data/icdar2013-Test/img_106.jpg'  
          lineGt = image_name.replace(".jpg", ".txt")
          if os.path.exists(lineGt):
              try:
                  word_gto = read_txt_gt(lineGt, multi_script, skip_slash = imageNo < 10000)
              except:
                  print("invalid gt: {0}".format(lineGt))
                  word_gto = []
                  #continue
          else:
              print("no gt: {0}".format(lineGt))
              continue
          
          im = cv2.imread(image_name)
          #image_size = [im.shape[1] / 2 / 32 * 32, im.shape[0] / 2 / 32 * 32] 
          #while image_size[0] > 608 or image_size[1] > 608:
          #    image_size[0] /= 1.5
          #    image_size[1] /= 1.5   
          #    image_size[0] = int(image_size[0]) /32 * 32
          #    image_size[1] = int(image_size[1]) /32 * 32
                
          
          scaled = cv2.resize(im, (image_size[0], image_size[1]))
          if self.gray_scale:
              scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY )
              scaled = scaled.reshape((scaled.shape[0], scaled.shape[1], 1))
          
          gto_out = self.transform_boxes(im, scaled, word_gto)
          scales.append((image_size[0] / float(scaled.shape[1]), image_size[1] / float(scaled.shape[0])  ))
          
          images.append(scaled)
          gt.append(gto_out)
          gt_orig.append(word_gto)
          imageso.append(im)
          
          images_read += 1
          names.append(image_name)
          
          if images_read >= count:
              break
          
      return (images, gt, scales, names, imageso, gt_orig), image_size
      
  def reset(self):
      self.counter = 0
      
  def has_next(self, set_name):
      image_list = self.data[set_name]
      return self.counter < (len(image_list)) 
  
  
class DataLoaderOCR(object):
  '''
  classdocs
  '''
  def __init__(self, sets, root_dir = '/'):
      '''
      Constructor
      '''
      
      self.data = {}
      for key in sets.keys():
          list_file = sets[key]
          images = []
          with open(list_file, "r") as ins:
              for line in ins:
                  images.append(line.strip())
                  
          self.data[key] = images
          self.counter = 0
          self.it = 0
          self.step = 50
          self.crop_ratio = 0.3
          
      self.root_dir = root_dir
  
      
  def get_image(self, set_name, count):
      
      image_list = self.data[set_name]
      if self.it % 30 == 0:
          self.step += 1
      self.it += 1    
      self.step = min(self.step, len(image_list) - 1) 
      while True:    
          
          imageNo = random.randint(0, len(image_list) - 1)
          image_name = image_list[imageNo]
          try:
              if image_name[0] != '/':
                  image_name = '{0}/{1}'.format(self.root_dir, image_name)
              
              spl = image_name.split(" ")
              image_name = spl[0].strip()
              gt_txt = ''
              if len(spl) > 1:
                  gt_txt = spl[1].strip()
                  if gt_txt[0] == '"':
                      gt_txt = gt_txt[1:-1]
                      
  
              if image_name[len(image_name) - 1] == ',':
                  image_name = image_name[0:-1]    
              
              gt_txt = [item - 28 if item < 177 else 100 for item in gt_txt]
              
              im = cv2.imread(image_name)
              print(image_name)
              return im, gt_txt, image_name
          except: 
              print(image_name)
            
        
        
        
    
       
    
