"""Atomic data-only checkpoints for model, optimizer, scheduler and RNG state."""
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from .serialization import _read_tensor, MAX_NODES

MAX_CHECKPOINT_BYTES=2*1024**3


def save_state(state,path):
    arrays=[]
    def encode(value):
        if isinstance(value,np.ndarray):
            if value.dtype!=np.float32: raise ValueError('checkpoint tensors must be float32')
            arrays.append(value);return ['array',len(arrays)-1,list(value.shape)]
        if isinstance(value,dict): return ['dict',[[str(k),encode(v)] for k,v in value.items()]]
        if isinstance(value,(tuple,list)): return ['list',[encode(v) for v in value]]
        if isinstance(value,np.generic): value=value.item()
        if value is None or type(value) in (bool,int,float,str): return ['value',value]
        raise TypeError(f'unsupported checkpoint value {type(value)}')
    manifest={'format':'munet-training-state','version':1,'state':encode(state)}
    encoded=json.dumps(manifest,allow_nan=False,separators=(',',':')).encode()
    if len(arrays)>MAX_NODES or len(encoded)+sum(a.nbytes+128 for a in arrays)>MAX_CHECKPOINT_BYTES: raise ValueError('checkpoint exceeds size limit')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f'.{path.name}.',dir=path.parent);os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED) as z:
            z.writestr('manifest.json',encoded)
            for i,a in enumerate(arrays):
                with z.open(f'tensors/{i}.npy','w',force_zip64=True) as out: np.lib.format.write_array(out,np.ascontiguousarray(a) if a.ndim else a,allow_pickle=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_state(path):
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or len(names)>MAX_NODES+1 or sum(i.file_size for i in infos)>MAX_CHECKPOINT_BYTES: raise ValueError('invalid or oversized checkpoint')
        if 'manifest.json' not in names or z.getinfo('manifest.json').file_size>32*1024**2: raise ValueError('missing or oversized manifest')
        m=json.loads(z.read('manifest.json'))
        if m.get('format')!='munet-training-state' or m.get('version')!=1: raise ValueError('unsupported training checkpoint')
        used={'manifest.json'}
        def decode(value,depth=0):
            if depth>64 or not isinstance(value,list) or not value: raise ValueError('invalid checkpoint tree')
            kind=value[0]
            if kind=='array' and len(value)==3 and type(value[1]) is int and value[1]>=0:
                name=f'tensors/{value[1]}.npy'
                if name in used: raise ValueError('duplicate checkpoint tensor')
                used.add(name);return _read_tensor(z.read(name),value[2]).copy()
            if len(value)!=2: raise ValueError('invalid checkpoint entry')
            if kind=='value' and (value[1] is None or type(value[1]) in (bool,int,float,str)): return value[1]
            if kind=='list': return [decode(v,depth+1) for v in value[1]]
            if kind=='dict':
                result={}
                for k,v in value[1]:
                    if not isinstance(k,str) or k in result: raise ValueError('invalid checkpoint dictionary')
                    result[k]=decode(v,depth+1)
                return result
            raise ValueError('unknown checkpoint entry')
        result=decode(m['state'])
        if used!=set(names): raise ValueError('unexpected checkpoint entries')
        return result
