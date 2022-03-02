#-*- coding: UTF-8 -*-  
#!/usr/bin/env python

from pathlib import Path
cur_file_path = Path(__file__).absolute()
workdir = Path(cur_file_path).parent.parent

import argparse
import configparser
import cv2
import numpy as np
import os
import signal
import threading
from gevent import monkey
from gevent.pywsgi import WSGIServer
monkey.patch_all()
from flask import Flask,request,jsonify
from detection import MeterDetection
from log import detlog

            
def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    os._exit(0)
    
app = Flask(__name__)
@app.route('/meter',methods=['POST'])
def demo():   
    name=request.form['filename']
    logger.info(name)
    img=cv2.imread(name)
    predict=model(img).tolist()
    print(predict)
    res={'predict':predict}
    return res
    
def parse_args():
    parser = argparse.ArgumentParser(description='Flask demo')
    parser.add_argument('--gpu', dest='gpu',type=int,default=0)
    parser.add_argument('--port',dest='port',type=int,default=1111)
    parser.add_argument('--gpuRatio',dest='gpuRatio',type=float,default=0.1)
    parser.add_argument('--host',dest='host',type=str,default='0.0.0.0')
    parser.add_argument('--logID',dest='logID',type=str,default='0')
    args = parser.parse_args()
    return args

def serv_start():
    global host, portNum
    print(host,portNum)
    logger.info('serv starting...')

    http_server = WSGIServer((host, portNum), app)
    http_server.serve_forever()

    logger.info('serv started')
       

if __name__ == '__main__':
    
    args = parse_args()
    portNum = args.port
    host = args.host
    cf=configparser.ConfigParser()
    cf.read('config.ini')
    modelname=Path(cf.get('common','model_path')).stem
    logfilename='mmm'
    logger=detlog(modelname,logfilename,args.logID)
    model=MeterDetection(args,cf,logger=logger)
    logger.info('instance of model created')
    threads = []
    t0 = threading.Thread(target=serv_start)
    threads.append(t0)
    t1=threading.Thread(target=model.model_restore)
    threads.append(t1)
    print('-*'*20)
    signal.signal(signal.SIGINT, signal_handler)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

