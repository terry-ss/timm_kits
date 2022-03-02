import abc

class Detection(abc.ABC):
    def __init__(self, args, cf):
        self.read_args(args)
        self.cf=cf
        
    @abc.abstractmethod
    def read_args(self, args):
        pass

    @abc.abstractmethod
    def model_restore(self):
        pass

    @abc.abstractmethod
    def warmup(self):
        # run model with a random input
        pass

    def __call__(self,im):
        return self.forward(im)
    
    @abc.abstractmethod
    def forward(self,im):
        """
        detect img
        """
        pass
  
    @abc.abstractmethod
    def process(self):
        """
        pre-porcess for input image.
        """
        pass
        
if __name__ == '__main__':
    pass
        

