import pandas as pd
import yaml
import sys
from pathlib import Path
from rich.progress import track
from rich.console import Console
from rich.text import Text
from importlib import machinery
_cur=Path(__file__).absolute().parent
machinery.SourceFileLoader('m_utils',str(_cur/'../libs/model_utils.py')).load_module()
from m_utils import get_labels

def csv2yaml(csv_path,yaml_path):
    df=pd.read_csv(csv_path)
    if not yaml_path.endswith('.yaml'):
        text=Text('invalid yaml path')
        text.stylize("bold red")
        console.print(text)
        exit()
    _dict={}
    classLabels=get_labels('data/labels_name.csv')
    num_classes=len(classLabels)
    for row in track(df.itertuples(),total=len(df)):
        filepath=row.filepath
        name=Path(filepath).name
        if name not in _dict.keys():
            data=list(row)[2:]
            if len(data)!=num_classes:
                text=Text(f'{filepath} is invalid')
                text.stylize("bold magenta")
                console.print(text)
                continue
            _dict[name]=data
        else:
            text=Text(f'{filepath} is reduplicate and skippd')
            text.stylize("bold magenta")
            console.print(text)
    with open(yaml_path,'w') as f:
        yaml.dump(_dict,f)
    
    
if __name__ == '__main__':
    csv_path=sys.argv[1];yaml_path=sys.argv[2]
    console = Console()
    csv2yaml(csv_path,yaml_path)
    