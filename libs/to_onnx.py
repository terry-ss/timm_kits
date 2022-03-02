import torch
import onnx
from pathlib import Path
import sys

def convert2onnx(model_path,input_size,out_path=None):
    if out_path is None:
        out_path=Path(model_path).with_suffix('.onnx')
    model=torch.load(model_path)
    out_path=export2onnx(model,input_size,out_path)
    return out_path
    

def export2onnx(model,input_size,onnx_path,if_check=True,if_simplify=True):
    model.eval()
    c,h,w=tuple(input_size)
    device='cuda' if torch.cuda.is_available() else 'cpu' 
    model.to(device)
    dummy_input = torch.randn(1,c,h,w).to(device)
    assert onnx_path.endswith('.onnx')

    torch.onnx._export(model,dummy_input,onnx_path,
        verbose=False)
    print('onnx exported')
    if if_check:
        onnx_path=check_onnx(onnx_path)
    if if_simplify:
        out_path=simplify_onnx(onnx_path)
        Path(onnx_path).unlink()
    else:
        out_path=onnx_path
    return out_path

def check_onnx(onnx_path):
    onnx_model=onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('valid onnx model')
    return onnx_path
  
def simplify_onnx(onnx_path):
    from onnxsim import simplify
    onnx_path=str(onnx_path)
    onnx_model=onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    out_path=onnx_path.replace('.onnx','_simp.onnx')
    onnx.save(model_simp,out_path)
    print('simplify done')
    return out_path
    
if __name__ == '__main__':
    # model_path,c,w,h=sys.path[1:5]
    # convert2onnx(model_path,c,w,h)
    simply_onnx('model/model_Number_aus_bin.onnx')
