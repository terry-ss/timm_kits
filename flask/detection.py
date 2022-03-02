import cv2
import torch
import os
import numpy as np
import onnxruntime
onnxruntime.set_default_logger_severity(3)
from pathlib import Path
from log import detlog
from base import Detection

class MeterDetection(Detection):
    def __init__(self,args,cf,**kwarg):
        super().__init__(args,cf)
        # self.read_args(args)
        # self.cf=cf
        self.model_path=Path(cf.get('common','model_path'))
        self.size=eval(cf.get('common','size'))
        os.environ['CUDA_VISIBLE_DEVICES']=self.gpu
        self.log=kwarg['logger']
        self.model_restore()
        
    def read_args(self, args):
        self.portNum  = args.port
        self.gpu    = str(args.gpu)
        #self.gpuRatio = args.gpuRatio
        self.host     = args.host
        self.logID    = args.logID
        
    def model_restore(self):
        self.log.info('===== model restore start =====')
        onnx_model_path=str(self.model_path)
        #print(onnx_model_path)
        if not Path(onnx_model_path).exists():
            raise FileNotFoundError(onnx_model_path)
        elif not onnx_model_path.endswith('.onnx'):
            raise RuntimeError('only support onnx model')
        ort_session = onnxruntime.InferenceSession(onnx_model_path)
        self.model=ort_session
        self.warmup()
        
        
    def warmup(self):
        h,w=self.size
        im=128*np.ones((h,w,3),dtype=np.uint8)
        self.forward(im)
        print('meter model warmup done')
        
    def forward(self,im):
        if im is None:
            print('img is None!')
            out=None
        else:
            img_tensor=self.process(im)
            out=self.model.run(None, img_tensor)[0].squeeze()
        self.log.info(out)
        return out
      
    def process(self,img):
        normSize=self.size
        img=cv2.resize(img, normSize)
        img=np.float32(img)
        inut_array=np.expand_dims(np.transpose(img,(2,0,1)),axis=0)
        ort_inputs = {self.model.get_inputs()[0].name: inut_array}
        return ort_inputs
