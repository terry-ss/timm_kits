import logging
import logging.handlers
import sys
from pathlib import Path
import datetime

_cur=Path(__file__).absolute().parent

class detlog(object):

    def __init__(self, model, filename,logID):
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        logdir=_cur/'logs'/model/f'{date}-{logID}'
        logdir.mkdir(parents=True, exist_ok=True)
        logpath=logdir/f'{filename}.log'
        logpath.touch()
        loglen = 1024*1024*1024

        self.handler = logging.handlers.RotatingFileHandler(logpath, maxBytes = loglen, backupCount = 10) # 实例化handler
        self.fmt = '%(asctime)s - %(message)s'
        #self.fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
        formatter = logging.Formatter(self.fmt)   # 
        self.handler.setFormatter(formatter)      # 
        self.logger = logging.getLogger('tst')    # 
        self.logger.addHandler(self.handler)           # 
        self.logger.setLevel(logging.DEBUG)

    def info(self,*msgs):
        #sysFrame = sys._getframe()
        #lineNo = sysFrame.f_back.f_lineno
        #funcName = sysFrame.f_back.f_code.co_name
        lineNo,funcName = self.get_cur_info()
        for msg in msgs:
            self.logger.debug('%s:%s - %s' % (funcName,lineNo,msg))

    def get_cur_info(self):
        try:
            raise Exception
        except Exception as e:
            #lineNo = e.__traceback__.tb_lineno
            #funcName = e.__traceback__.tb_frame.f_globals["__file__"]
            if e.__traceback__.tb_frame.f_back.f_back == None:
                lineNo = e.__traceback__.tb_frame.f_back.f_lineno
                funcName = e.__traceback__.tb_frame.f_back.f_code.co_name
            else:
                lineNo = e.__traceback__.tb_frame.f_back.f_back.f_lineno
                funcName = e.__traceback__.tb_frame.f_back.f_back.f_code.co_name
            return lineNo,funcName

    def debug(self,funcName,*msgs,depth=0):
        for msg in msgs:
            self.logger.debug('GOT ERROR: %s---->%s',funcName,msg)

    def recordTime(self,funcName,time):
        self.logger.debug('%s spend Time: %ss' % (funcName,time))
        
    def line(self,*msgs,filler="=",totLen=40):
        lineNo,funcName = self.get_cur_info()
        for msg in msgs:
            msgLen = len(msg)
            fillerLen = int((totLen - msgLen)/2) + (totLen - msgLen)%2
            lenMsg = filler * fillerLen + msg + filler * fillerLen
            self.logger.debug('%s:%s - %s' % (funcName,lineNo,lenMsg)) 
            
if __name__ == '__main__':
    print(type(_cur))