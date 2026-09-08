"""Static float32 ONNX 13..18 inference interchange; all dynamic math is native.

Integer shape constants are folded at import. Runtime indices/masks use exact
float32 integers (bounded to 2**24), as in the native detector. This is not a
promise of arbitrary INT64 arithmetic or dynamic-shape/control-flow support.
"""
import json
import tempfile
from pathlib import Path
import numpy as np
from . import _native, nn
from .core import Tensor, _Trace, _active, as_tensor, operation, cat, where
from .serialization import from_graph


class UnsupportedOperatorError(ValueError): pass


SIMPLE = {'Add':'add','Sub':'sub','Mul':'mul','Div':'div','Neg':'neg','MatMul':'matmul',
          'Relu':'relu','Sigmoid':'sigmoid','Exp':'exp','Log':'log','Sqrt':'sqrt','Identity':'identity',
          'Abs':'abs','Sign':'sign','Tanh':'tanh','Erf':'erf','Sin':'sin','Cos':'cos','Floor':'floor',
          'Min':'minimum','Max':'maximum','Less':'lt','LessOrEqual':'le','Equal':'eq',
          'Greater':'gt','GreaterOrEqual':'ge','Where':'where','Softplus':'softplus'}
SUPPORTED_ONNX = set(SIMPLE) | {'Constant','Gemm','Transpose','Reshape','Expand','ReduceSum','ReduceMean',
    'ReduceMax','Conv','MaxPool','AveragePool','GlobalAveragePool','BatchNormalization','LayerNormalization',
    'Softmax','LogSoftmax','GridSample','Resize','Slice','Concat','Gather','GatherElements','TopK','Cast',
    'Shape','Size','Unsqueeze','Squeeze','Split','Flatten','Clip','Pow','Not','And','Or','ConstantOfShape','Tile'}


def from_onnx(path, *, device='vulkan', fuse=True):
    import onnx
    from onnx import helper, numpy_helper, TensorProto as T
    from onnx.reference import ReferenceEvaluator
    model=onnx.load(str(path),load_external_data=False)
    if model.functions:
        from onnx.inliner import inline_local_functions
        model=inline_local_functions(model)
    if model.training_info or model.graph.sparse_initializer:
        raise UnsupportedOperatorError('ONNX training_info and sparse initializers are unsupported')
    if any(t.data_location==T.EXTERNAL for t in model.graph.initializer):
        raise UnsupportedOperatorError('save a self-contained ONNX model; external tensor data is unsupported')
    imports={x.domain:x.version for x in model.opset_import}
    if not 13<=imports.get('',0)<=18 or any(n.domain for n in model.graph.node):
        raise UnsupportedOperatorError(f'supported contract is default-domain ONNX opset 13..18; received {imports}')
    unsupported=[(n.name,n.op_type) for n in model.graph.node if n.op_type not in SUPPORTED_ONNX]
    if unsupported: raise UnsupportedOperatorError(f'unsupported ONNX nodes: {unsupported}')
    onnx.checker.check_model(model,full_check=True)
    ctx=_Trace();g=ctx.graph;values={};constants={};specs=[];input_values=[]
    initializers={x.name:x for x in model.graph.initializer}
    if set(initializers)&{v.name for v in model.graph.input}: raise UnsupportedOperatorError('overridable initializers are unsupported')
    def constant(name,a): constants[name]=np.asarray(a)
    def get(name):
        if name not in values:
            a=constants[name]
            if a.dtype not in (np.float32,np.int32,np.int64,np.bool_): raise UnsupportedOperatorError(f'unsupported dtype {a.dtype}')
            if a.dtype.kind in 'iu' and np.any(np.abs(a.astype(np.float64))>2**24): raise UnsupportedOperatorError('runtime index exceeds exact float32 integer range')
            values[name]=as_tensor(a)
        return values[name]
    def ints(name):
        a=constants[name]
        if a.dtype.kind not in 'iu': raise UnsupportedOperatorError('expected a constant integer shape/axes tensor')
        return a.reshape(-1).tolist()
    def pair(a):
        if len(a)!=2: raise UnsupportedOperatorError('only two spatial dimensions are supported')
        return tuple(a)
    def symmetric(attrs):
        p=attrs.get('pads',[0,0,0,0])
        if len(p)!=4 or p[:2]!=p[2:] or attrs.get('auto_pad',b'NOTSET') not in (b'NOTSET',b'VALID'):
            raise UnsupportedOperatorError('Conv/Pool require explicit symmetric spatial padding')
        return pair(p[:2])
    token=_active.set(ctx)
    try:
        for v in model.graph.input:
            t=v.type.tensor_type
            if t.elem_type!=T.FLOAT or any(not d.HasField('dim_value') or d.dim_value<=0 for d in t.shape.dim):
                raise UnsupportedOperatorError(f'input {v.name} needs a static positive float32 shape')
            shape=[d.dim_value for d in t.shape.dim];ident=g.leaf('input',v.name,shape)
            values[v.name]=Tensor(g,ident);specs.append(shape);input_values.append(ident)
        for name,v in initializers.items(): constant(name,numpy_helper.to_array(v))
        for node in model.graph.node:
            op=node.op_type;a={x.name:helper.get_attribute_value(x) for x in node.attribute};names=list(node.input)
            try:
                if op=='Constant':
                    if set(a)!={'value'}: raise UnsupportedOperatorError('Constant requires a tensor value')
                    constant(node.output[0],numpy_helper.to_array(a['value']));continue
                if op=='Shape':
                    shape=get(names[0]).shape
                    constant(node.output[0],np.asarray(shape[a.get('start',0):a.get('end',len(shape))],np.int64));continue
                if op=='Size': constant(node.output[0],np.asarray(np.prod(get(names[0]).shape),np.int64));continue
                # Evaluate only constant subgraphs; no reference evaluator is retained
                # in the resulting program, and runtime inputs never enter this path.
                if all(not n or n in constants for n in names):
                    results=ReferenceEvaluator(node,opsets={'':imports['']}).run(None,{n:constants[n] for n in names if n})
                    for n,v in zip(node.output,results): constant(n,v)
                    continue
                x=get(names[0]);result=None
                if op in SIMPLE: result=operation(SIMPLE[op],*[get(n) for n in names])
                elif op=='Gemm':
                    b=get(names[1]);y=x.T if a.get('transA',0) else x;b=b.T if a.get('transB',0) else b
                    result=(y@b)*a.get('alpha',1.)
                    if len(names)>2 and names[2]: result=result+get(names[2])*a.get('beta',1.)
                elif op=='Transpose': result=x.permute(a.get('perm',list(reversed(range(x.ndim)))))
                elif op in ('Reshape','Expand'):
                    shape=ints(names[1])
                    if op=='Reshape':
                        if not a.get('allowzero',0): shape=[x.shape[i] if s==0 else s for i,s in enumerate(shape)]
                        result=x.reshape(shape)
                    else: result=x.expand(np.broadcast_shapes(x.shape,shape))
                elif op in ('ReduceSum','ReduceMean','ReduceMax'):
                    axes=ints(names[1]) if len(names)>1 and names[1] else a.get('axes',[])
                    if not axes and a.get('noop_with_empty_axes',0): result=x
                    else:
                        axes=axes or list(range(x.ndim));method={'ReduceSum':x.sum,'ReduceMean':x.mean,'ReduceMax':x.amax}[op]
                        result=method(axes,keepdim=bool(a.get('keepdims',1)))
                elif op=='Conv':
                    result=nn.functional.conv2d(x,get(names[1]),get(names[2]) if len(names)>2 and names[2] else None,
                        pair(a.get('strides',[1,1])),symmetric(a),pair(a.get('dilations',[1,1])),a.get('group',1))
                elif op in ('MaxPool','AveragePool'):
                    if len(node.output)!=1 or a.get('dilations',[1,1])!=[1,1] or a.get('storage_order',0): raise UnsupportedOperatorError('pool indices/dilation/storage_order unsupported')
                    kwargs=dict(kernel_size=pair(a['kernel_shape']),stride=pair(a.get('strides',[1,1])),padding=symmetric(a),ceil_mode=bool(a.get('ceil_mode',0)))
                    if op=='AveragePool': kwargs['count_include_pad']=bool(a.get('count_include_pad',0))
                    result=(nn.functional.max_pool2d if op=='MaxPool' else nn.functional.avg_pool2d)(x,**kwargs)
                elif op=='GlobalAveragePool': result=x.mean(tuple(range(2,x.ndim)),keepdim=True)
                elif op=='BatchNormalization':
                    if a.get('training_mode',0) or len(node.output)!=1: raise UnsupportedOperatorError('only inference BatchNormalization is imported')
                    scale,bias,mean,var=[get(n).reshape(1,-1,1,1) for n in names[1:]]
                    result=(x-mean)/(var+a.get('epsilon',1e-5)).sqrt()*scale+bias
                elif op=='LayerNormalization':
                    if len(node.output)!=1 or a.get('stash_type',T.FLOAT)!=T.FLOAT: raise UnsupportedOperatorError('LayerNormalization auxiliary outputs/dtypes unsupported')
                    axis=a.get('axis',-1)%x.ndim;axes=tuple(range(axis,x.ndim));center=x-x.mean(axes,True)
                    result=center/(center.square().mean(axes,True)+a.get('epsilon',1e-5)).sqrt()*get(names[1])
                    if len(names)>2 and names[2]: result=result+get(names[2])
                elif op in ('Softmax','LogSoftmax'):
                    axis=a.get('axis',-1)
                    if op=='Softmax': result=x.softmax(axis)
                    else:
                        z=x-x.amax(axis,True);result=z-z.exp().sum(axis,True).log()
                elif op=='GridSample':
                    if a.get('mode',b'bilinear')!=b'bilinear' or a.get('padding_mode',b'zeros')!=b'zeros' or a.get('align_corners',0): raise UnsupportedOperatorError('GridSample requires bilinear/zeros/align_corners=False')
                    result=nn.functional.grid_sample(x,get(names[1]))
                elif op=='Resize':
                    if a.get('mode',b'nearest')!=b'nearest' or a.get('coordinate_transformation_mode',b'half_pixel')!=b'asymmetric' or a.get('nearest_mode',b'round_prefer_floor')!='floor'.encode() or a.get('antialias',0) or a.get('axes',list(range(x.ndim)))!=list(range(x.ndim)): raise UnsupportedOperatorError('Resize requires asymmetric/floor nearest spatial interpolation')
                    shape=ints(names[3]) if len(names)>3 and names[3] else [int(d*s) for d,s in zip(x.shape,constants[names[2]])]
                    if shape[:2]!=list(x.shape[:2]): raise UnsupportedOperatorError('Resize batch/channel changes unsupported')
                    result=nn.functional.interpolate(x,size=shape[-2:],mode='nearest')
                elif op=='Slice':
                    starts,ends=ints(names[1]),ints(names[2]);axes=ints(names[3]) if len(names)>3 and names[3] else list(range(len(starts)));steps=ints(names[4]) if len(names)>4 and names[4] else [1]*len(starts)
                    key=[slice(None)]*x.ndim
                    for axis,start,end,step in zip(axes,starts,ends,steps): key[axis]=slice(start,end,step)
                    result=x[tuple(key)]
                elif op=='Concat': result=cat([get(n) for n in names],a.get('axis',0))
                elif op in ('Gather','GatherElements'):
                    axis=a.get('axis',0);idx=get(names[1]);idx=where(idx<0,idx+x.shape[axis],idx)
                    result=x.take(idx,axis) if op=='Gather' else x.gather(axis,idx)
                elif op=='TopK':
                    k=ints(names[1])[0];axis=a.get('axis',-1)
                    vals,idx=(x if a.get('largest',1) else -x).topk(k,axis)
                    result=(vals if a.get('largest',1) else -vals,idx)
                elif op=='Cast':
                    target=a['to']
                    if target==T.FLOAT: result=x
                    elif target==T.BOOL: result=(x.eq(0)).eq(0)
                    elif target in (T.INT32,T.INT64): result=where(x>=0,x.floor(),-(-x).floor())
                    else: raise UnsupportedOperatorError('unsupported Cast dtype')
                elif op in ('Unsqueeze','Squeeze'):
                    axes=ints(names[1]) if len(names)>1 else a.get('axes',[])
                    if op=='Unsqueeze':
                        result=x
                        for axis in sorted(d%(x.ndim+len(axes)) for d in axes): result=result.unsqueeze(axis)
                    else:
                        if any(x.shape[d]!=1 for d in axes): raise UnsupportedOperatorError('Squeeze axis is not size one')
                        result=x.reshape([s for i,s in enumerate(x.shape) if i not in [d%x.ndim for d in axes]]) if axes else x.squeeze()
                elif op=='Split':
                    axis=a.get('axis',0)
                    sizes=ints(names[1]) if len(names)>1 and names[1] else [x.shape[axis]//len(node.output)]*len(node.output)
                    result=x.split(sizes,axis)
                elif op=='Flatten':
                    axis=a.get('axis',1);axis=axis+x.ndim if axis<0 else axis;result=x.reshape(int(np.prod(x.shape[:axis])),int(np.prod(x.shape[axis:])))
                elif op=='Clip': result=x.clamp(min=float(constants[names[1]]) if len(names)>1 and names[1] else None,max=float(constants[names[2]]) if len(names)>2 and names[2] else None)
                elif op=='Pow':
                    exponent=constants[names[1]]
                    if exponent.size!=1: raise UnsupportedOperatorError('Pow requires scalar constant exponent')
                    exponent=float(exponent);result=x**(int(exponent) if exponent.is_integer() else exponent)
                elif op=='Not': result=x.eq(0)
                elif op=='And': result=(x*get(names[1])).eq(0).eq(0)
                elif op=='Or': result=(x.eq(0)*get(names[1]).eq(0)).eq(0)
                elif op=='Tile': result=x.repeat(ints(names[1]))
                else: raise UnsupportedOperatorError(f'no native lowering for {op}')
                outputs=list(result) if isinstance(result,(list,tuple)) else [result]
                if len(outputs)!=len(node.output): raise UnsupportedOperatorError('operator output count mismatch')
                for n,v in zip(node.output,outputs): values[n]=v
            except (KeyError,IndexError,ValueError,TypeError) as exc:
                raise UnsupportedOperatorError(f'cannot lower {node.name or node.output[0]} ({op}): {exc}') from exc
        outputs=[]
        for v in model.graph.output:
            if v.type.tensor_type.elem_type!=T.FLOAT: raise UnsupportedOperatorError('public runtime outputs must be float32; indices are represented as exact floats')
            result=get(v.name);declared=v.type.tensor_type.shape.dim
            if len(declared)!=result.ndim or any(d.HasField('dim_value') and d.dim_value!=s for d,s in zip(declared,result.shape)): raise ValueError(f'declared output shape differs: {v.name}')
            outputs.append(result.value)
    finally: _active.reset(token)
    plan=_native.Plan(g,outputs,[],fuse);feeds=[input_values.index(i) for i in plan.inputs]
    program=from_graph(g,outputs,[],specs,feeds,len(outputs)==1,device=device,fuse=fuse)
    meta={p.key:p.value for p in model.metadata_props}
    if 'munet.output_tree' in meta:
        from .serialization import validate_output_tree
        program._output_tree=validate_output_tree(json.loads(meta['munet.output_tree']),len(outputs))
    return program


def to_onnx(program,path):
    import onnx
    from onnx import helper as h,numpy_helper as nh,TensorProto as T
    if program._plan is None: raise ValueError('call the compiled program before exporting')
    if program._plan.updates: raise UnsupportedOperatorError('ONNX export is an inference graph; compile an eval model separately')
    with program._lock:
        for p,value in program._parameters.items():
            if p._version!=program._seen_versions[p]: program._plan.write(value,np.ascontiguousarray(p.numpy()));program._seen_versions[p]=p._version
        nodes=program._plan.graph.nodes(data=False);live=set(program._plan.outputs);pending=list(live)
        while pending:
            i=pending.pop()
            # Rank implementation is replaced by standard TopK during export.
            for parent in nodes[i]['inputs'][:1] if nodes[i]['op']=='topk_indices' else nodes[i]['inputs']:
                if parent not in live: live.add(parent);pending.append(parent)
        names=[f'v{i}' for i in range(len(nodes))];inputs=[];initializers=[];ops=[]
        def const(value,dtype=np.float32):
            name=f'k{len(initializers)}';initializers.append(nh.from_array(np.asarray(value,dtype=dtype),name));return name
        def emit(kind,args,out=None,**attrs):
            out=out or f't{len(ops)}';ops.append(h.make_node(kind,args,[out],name=f'node_{len(ops)}',**attrs));return out
        simple={v:k for k,v in SIMPLE.items()};simple['detach']='Identity'
        for i,n in enumerate(nodes):
            if i not in live: continue
            op,name=n['op'],names[i];a=n['attrs'];args=[names[j] for j in n['inputs']]
            if op=='input': inputs.append(h.make_tensor_value_info(name,T.FLOAT,n['shape']));continue
            if op in ('parameter','constant'): initializers.append(nh.from_array(program._plan.read(i),name));continue
            if op in ('lt','le','eq','gt','ge'):
                emit('Cast',[emit(simple[op],args)],name,to=T.FLOAT)
            elif op=='where': emit('Where',[emit('Cast',[args[0]],to=T.BOOL),*args[1:]],name)
            elif op in simple: emit(simple[op],args,name)
            elif op in ('transpose','permute'): emit('Transpose',args,name,perm=[1,0] if op=='transpose' else a)
            elif op in ('reshape','broadcast','sum','max'):
                emit({'reshape':'Reshape','broadcast':'Expand','sum':'ReduceSum','max':'ReduceMax'}[op],args+[const(a,np.int64)],name,**({'keepdims':1,'noop_with_empty_axes':1} if op in ('sum','max') else {}))
            elif op=='gelu':
                z=emit('Erf',[emit('Mul',[args[0],const(2**-.5)])]);z=emit('Add',[z,const(1.)]);emit('Mul',[emit('Mul',[args[0],const(.5)]),z],name)
            elif op=='conv2d': emit('Conv',args,name,strides=a[:2],pads=a[2:4]*2,dilations=a[4:6],group=a[6])
            elif op in ('max_pool2d','avg_pool2d'):
                emit('MaxPool' if op=='max_pool2d' else 'AveragePool',args,name,kernel_shape=a[:2],strides=a[2:4],pads=a[4:6]*2,ceil_mode=a[6],**({'count_include_pad':a[7]} if op=='avg_pool2d' else {}))
            elif op=='grid_sample': emit('GridSample',args,name,mode='bilinear',padding_mode='zeros',align_corners=0)
            elif op=='resize_nearest': emit('Resize',[args[0],'','',const(n['shape'],np.int64)],name,mode='nearest',coordinate_transformation_mode='asymmetric',nearest_mode='floor')
            elif op=='concat': emit('Concat',args,name,axis=a[0])
            elif op=='slice':
                r=len(a)//3;ends=[s+size*step for s,size,step in zip(a[:r],a[r:2*r],a[2*r:])];ends=[-(2**63) if step<0 and end<0 else end for end,step in zip(ends,a[2*r:])]
                emit('Slice',args+[const(a[:r],np.int64),const(ends,np.int64),const(list(range(r)),np.int64),const(a[2*r:],np.int64)],name)
            elif op=='topk_indices':
                ops.append(h.make_node('TopK',[args[0],const([a[1]],np.int64)],[f'values_{i}',f'indices_{i}'],axis=a[0],largest=1,sorted=1,name=f'node_{len(ops)}'))
                emit('Cast',[f'indices_{i}'],name,to=T.FLOAT)
            elif op in ('gather','take'):
                # Native padding indices return zero; standard ONNX Gather permits
                # negatives. Sanitize indices and mask the result to preserve padding.
                idx=args[1];dim=nodes[n['inputs'][0]]['shape'][a[0]]
                valid=emit('And',[emit('GreaterOrEqual',[idx,const(0.)]),emit('Less',[idx,const(dim)])])
                valid=emit('And',[valid,emit('Equal',[idx,emit('Floor',[idx])])])
                safe=emit('Cast',[emit('Where',[valid,idx,const(0.)])],to=T.INT64)
                y=emit('GatherElements' if op=='gather' else 'Gather',[args[0],safe],axis=a[0])
                if op=='take':
                    r=len(nodes[n['inputs'][0]]['shape']);axis=a[0]%r;ish=nodes[n['inputs'][1]]['shape'];valid=emit('Reshape',[valid,const([1]*axis+ish+[1]*(r-axis-1),np.int64)])
                emit('Where',[valid,y,const(0.)],name)
            else: raise UnsupportedOperatorError(f'no standard inference ONNX export for native op {op}')
        outputs=[h.make_tensor_value_info(names[i],T.FLOAT,nodes[i]['shape']) for i in program._plan.outputs]
        graph=h.make_graph(ops,'MuNet RT-DETR',inputs,outputs,initializers)
        model=h.make_model(graph,producer_name='munet-nn',opset_imports=[h.make_opsetid('',18)],ir_version=8)
        if getattr(program,'_output_tree',None): h.set_model_props(model,{'munet.output_tree':json.dumps(program._output_tree)})
        onnx.checker.check_model(model,full_check=True);onnx.save_model(model,str(path))


def from_torch(model,example_inputs,*,device='vulkan',fuse=True):
    """Import an eval-mode torch module through modern ONNX export (no Torch runtime)."""
    import torch
    if model.training: raise ValueError('call model.eval() before importing an inference model')
    if not isinstance(example_inputs,tuple): example_inputs=(example_inputs,)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'model.onnx'
        torch.onnx.export(model,example_inputs,str(path),dynamo=True,opset_version=18,external_data=False)
        return from_onnx(path,device=device,fuse=fuse)
