# DeepTextSpotter

### DeepTextSpotter: An End-to-End Trainable Scene Text Localization and Recognition Framework

The implementation of [DeepTextSpotter](https://drive.google.com/open?id=0B8SUcdkLTcuTSmo4T2ozMWtDaUU): An End-to-End Trainable Scene Text Localization and Recognition Framework :  ICCV 2017.  

## Requirements
  - python2.7
  - opencv 3.x with python bindings

## Installation

1. Get the proper version of [caffe](https://github.com/MichalBusta/caffe.git)
  - or take required layers: Transpose, Reorg, Region, CTCLoss  

  - CTCLoss is warp around - https://github.com/baidu-research/warp-ctc - nice implementation, thanks! 

```
git clone https://github.com/MichalBusta/caffe.git
cd caffe
git checkout darknet

```

2. build caffe
 
```
mkdir Release 
cd Release 
cmake -D CMAKE_BUILD_TYPE=Release -D BLAS=Open -D BUILD_SHARED_LIBS=Off ..
make 
make install (optionally)
```

3. build project
```
cd "SOURCE dir" 
mkdir build
cd build
cmake ..
make 
```

## Download models

RPN: https://drive.google.com/open?id=0B8SUcdkLTcuTZjRHeUpjdzhmbFE
OCR: https://drive.google.com/open?id=0B8SUcdkLTcuTMmI0TS1uNDJaZGs

the paths are hard-coded, models shoud be at models subdirectory

## Run webcam demo

```
python2 demo.py
```

## Notes:
 - The provided RPN model is tiny version of full "YOLOv2" detector (= demo runs at 7 fps on 1GB Nvidia GPU) 
 - For decoding final output, we provide just greedy and dictionary based prefix decoding
 
 
