"""Evaluate named checkpoints on exactly the same images and optical sampling grid."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from doe.data import QuartetDataset
from doe.engine import load_model,setup,sync
from doe.model import gerchberg_saxton
from doe.results import phase_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/textures_v1')
    p.add_argument('--split',default='test',choices=('validation','test'))
    p.add_argument('--checkpoint',action='append',required=True,help='LABEL=PATH')
    p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    device=setup(device='cpu',threads=args.threads)
    models={};checkpoints={};config=None
    for value in args.checkpoint:
        name,path=value.split('=',1)
        if name in models:raise ValueError('duplicate label')
        model,_=load_model(path,device)
        spec=json.dumps(model.optics.config.to_dict(),sort_keys=True)
        if config is not None and spec!=config:raise ValueError('cannot compare different optical sampling grids as a model gain')
        config=spec;models[name]=model
        checkpoints[name]={'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'architecture':model.architecture()}
    dataset=QuartetDataset(args.data,args.split)
    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=0)
    optics=next(iter(models.values())).optics
    operations={name:model for name,model in models.items()}
    operations['gs_80']=lambda x:gerchberg_saxton(optics,x,80)
    records={name:[] for name in operations};seconds={name:0.0 for name in operations}
    with torch.no_grad():
        warm=next(iter(loader)).to(device)
        for fn in operations.values():fn(warm)
        for targets in loader:
            targets=targets.to(device)
            for name,fn in operations.items():
                sync(device);started=time.perf_counter();phase=fn(targets);sync(device)
                seconds[name]+=time.perf_counter()-started
                records[name].extend(phase_metrics(optics,phase,targets))
    summary={name:{'count':len(values),'pearson':float(np.mean([v['pearson'] for v in values])),
                   'ssim':float(np.mean([v['ssim'] for v in values])),
                   'seconds_per_quartet_batched':seconds[name]/len(values)} for name,values in records.items()}
    report={'split':args.split,'sample_count':len(dataset),'threads':args.threads,'batch_size':args.batch_size,
            'optics':json.loads(config),'checkpoints':checkpoints,'summary':summary,'per_sample':records}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
