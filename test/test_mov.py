from tinygrad.tensor import Tensor
import torch
import numpy as np
import random
import logging

logging.basicConfig(level=logging.WARNING, format='%(message)s')

def not_strided_2d(x,sh,st,off):
    a,b = x.shape
    rowstoadd = 0
    while 1:
        if b*(a+rowstoadd)%st[0]==0: break
        rowstoadd+=1
    padargs, kwargs = ((0,rowstoadd),(0,0)), dict(mode='constant', constant_values=0)
    x = pad(x, padargs, **kwargs)
    x = reshape(x, (x.size//st[0],st[0]))
    # pad cols
    o = st[1]-1
    colstoadd = (st[1]*sh[1]-o)-st[0]+off
    if colstoadd > 0:
        padargs,kwargs = ((0,0),(0,colstoadd)), dict(mode=pad_col_staggered,A=x,i=1,colstoadd=colstoadd,axis=1)
        x = pad(x,padargs,**kwargs)
        assert(st[1]*sh[1]-o)+off == len(x[0])
    # shrink
    x = shrink(x, ((0,sh[0]), (off,st[1]*sh[1]-o+off)))
    # stride
    x = stride(x, (1,st[1]))
    return x

def size(x):
    if isinstance(x, np.ndarray): return x.size
    assert isinstance(x, torch.Tensor)
    return x.nelement()
'''MovememntOps.SHRINK'''
def shrink(x, arg):
    _shrink = lambda x, arg: x[tuple(slice(p[0], p[1], None) for p in arg)]
    return _shrink(x,arg)

'''MovememntOps.STRIDE'''
def stride(x, arg):
    _stride = lambda x, arg: x[tuple(slice(None, None, i) for i in arg)]
    return _stride(x,arg)

'''MovementOps.STRIDE'''
def reshape(x,arg):
    _reshape = lambda x,arg: x.reshape(arg)
    return _reshape(x,arg)

'''MovementOps.PAD'''
def pad(x,args, **kwargs):
    return np.pad(x,args,**kwargs)

'''MovementOps.PERMUTE'''
def permute(x,order):
    return x.transpose(order)

def test_custom_pad():
    A = np.array([[1, 2], [-3, -4], [5, 6]])
    colstoadd = 1  # Example; you can adjust as needed
    kwargs =dict(i=1, colstoadd=colstoadd)
    B = np.pad(A, ((0,0),(0,1)), mode=pad_col_staggered, i=1, colstoadd=1, axis=1)

def construct_tests(n):
    tests = []
    for i in range(n):
        a,b = random.randint(2,10), random.randint(2,10)
        x = torch.arange(a*b).reshape(a,b)
        sh = (random.randint(2,a), random.randint(2,b))
        st_0 = random.randint(1,(a*b-sh[1])//sh[0])
        st_1 = random.randint(1,(a*b-sh[0]*st_0)//sh[1])
        st = (range(1,st_0+1), range(1,st_1+1))
        off = random.randint(0,a*b-sh[0]*st_0-sh[1]*st_1)
        print(f'** MADE TEST {i} **')
        print(f'x {x.shape} sh {sh} st {st} off {off}')
        tests.append((x,sh,st,off))
    return tests 

def construct_tests_nd(n):
    tests = []
    for i in range(n):
        d = random.randint(2, 5)
        sh_tensor = [random.randint(2, 5) for _ in range(d)]
        x = torch.arange(torch.prod(torch.tensor(sh_tensor))).reshape(*sh_tensor)
        sh = tuple(random.randint(1, dim) for dim in sh_tensor[:-1])  # Exclude the last dimension for the desired shape
        st = []
        for j in range(len(sh)):
            stride_value = random.randint(1, 5)
            st.append(stride_value)
        off = random.randint(1,size(x))
        try:
            torch.as_strided(x,sh,tuple(st),off)
        except:
            continue
        print(f'** MADE TEST {i} **')
        print(f'x {x.shape} sh {sh} st {tuple(st)} off {off}')
        tests.append((x, sh, tuple(st), off))
    
    return tests 

def run_tests(tests, f, offset=True):
    failed, faulty_tests = [], []
    t = 0
    for i,test in enumerate(tests):
        print(f'** TEST {i} **')
        x,sh,st,o = test
        off = 0 if offset else o
        flip = 0
        a = torch.as_strided(x,sh,st,off)
        b = f(x, sh, st, off)
        if flip:
            b = permute(b, tuple([-i for i in range(len(b.shape))])) 
        b = torch.tensor(b)
        try:
            assert torch.equal(a,b)
        except Exception as e:
            print(f'** failing st {st} sh {sh} off {off} **')
            print(a)
            print(b)
            failed.append((x,sh,st,off,a,b))
            t+=1
            continue
        print(f'** passing st {st} sh {sh} off {off} **')
        t+=1
    print('** ALL FAILED **')
    for fail in failed:
        x,sh,st,off,target,pred = fail
        print(f'** failing x {x.shape} sh {sh} st {st} off {off}**\n')
        print(target)
        print(pred)
        print()
    if DEBUG >= 1:
        print('** ALL FAULTY TESTS **')
        for fault in faulty_tests:
            x,sh,st,off = fault
            print(f'x {x.shape}, sh {sh} st {st} off {off}')
    print(f'** {len(failed)} FAILED {t-len(failed)} PASSED {100*len(failed)/t:.2f}% FAIL RATE')

def pad_col_staggered(vector, pad_width, iaxis, kwargs):
    # Extract the number of elements to pad before and after the current axis
    A, i, colstoadd, axis = kwargs['A'],kwargs['i'],kwargs['colstoadd'],kwargs['axis']
    if axis != iaxis: return
    kwargs['i'] += 1
    _, pad_after = pad_width
    if colstoadd <= A.shape[axis]:
        vector[-pad_after:]= A[i][:colstoadd] if i<len(A) else 0
    else:
        toadd,added,j = [],0,i
        while added < colstoadd:
            if j>=A.shape[axis-1]:
                toadd.extend([0]*(colstoadd-added))
                break
            toadd.extend(A[j][:colstoadd-added])
            added += len(A[j])
            j+=1
        vector[-pad_after:] = toadd 
    if DEBUG >= 1:
        print(f'i {i}')
        print(vector)

def get_zeros(x,st,d):
    zeros = 0
    while 1:
        if x.shape[d+1]*(x.shape[d]+zeros)%(st[d])==0:break
        zeros+=1
    return zeros

def multipad(vector,pad_width,iaxis,kwargs):
    A,padding,axis,i,pad_after = kwargs['A'],kwargs['padding'],kwargs['axis'],kwargs[f'i{kwargs["axis"]-1}'],pad_width[1]
    if not padding or iaxis != axis: return
    toadd = []
    while len(toadd) < padding:
        if i >= A.shape[axis-1]:
            zeros = np.zeros(tuple([padding-len(toadd)]+[1 for _ in range(len(vector.shape)-1)]))
            toadd.extend(zeros)
            break
        pre = [slice(kwargs[f'i{k}'],kwargs[f'i{k}']+1) for k in range(0,axis-1)]
        collected = A[tuple(pre)+(slice(i,i+1),)+(slice(padding-len(toadd)),)]
        ones = tuple(np.where(np.array(collected.shape)==1)[0])
        if max(collected.shape) == 1:
            ones = ones[:-1]
        collected = collected.squeeze(ones)
        if len(collected)>0:
            toadd.extend(collected)
        i+=1
    vector[-pad_after:] = toadd
    def update_idx(ax,kwargs,d=0):
        if f'i{ax-2}' not in kwargs: return False
        if kwargs[f'i{ax-1}']%A.shape[ax-1]==0:
            kwargs[f'i{ax-1}']=1 if d==0 else 0
            kwargs[f'i{ax-2}']+=1
            update_idx(ax-1,kwargs,d+1)
            return True 
        return False
    if not update_idx(axis,kwargs):
        kwargs[f'i{axis-1}']+=1

def not_strided(x, sh, st, off):
    dims = len(sh)-1
    if dims == 0: return reshape(not_strided(x,sh+(1,),st+(1,),off),sh)
    # process all axis except last 
    for d in range(dims):
        if d > 0: off = 0
        # add reshape padding to fix shape
        zeros = get_zeros(x,st,d)
        if zeros > 0:
            padargs = [(0, zeros if i==d else 0) for i in range(len(x.shape))]
            kwargs = dict(mode='constant', constant_values=0)
            x = pad(x, padargs, **kwargs)

        # adjust for st[0] by setting x.sh[1] = st[0]
        reshapeargs = [x.shape[i] for i in range(d)]+[(size(x) if d==0 else x.shape[d])//st[d],st[d]]+[1 for _ in range(dims-1-d)]
        assert len(reshapeargs) == len(sh)
        x = reshape(x, tuple(reshapeargs))

        # add wrap padding
        padding = (st[d+1]*sh[d+1]-(st[d+1]-1))-x.shape[d+1]+off if d==dims-1 else size(x)-x.shape[d+1]+off
        if padding > 0:
            padargs = tuple([(0, padding if i==d+1 else 0) for i in range(len(sh))])
            axisindices = {f'i{ax}':0 if ax!=d else 1 for ax in range(d+1)}
            kwargs = dict(mode=multipad, A=x, padding=padding, axis=d+1, **axisindices) 
            x = pad(x, padargs, **kwargs)

        assert len(sh) == len(x.shape) 
        shrinkargs = [(0,sh[i]) if i==d else (off if i==d+1 else 0, x.shape[i]) for i in range(len(sh))]
        x = shrink(x, tuple(shrinkargs))

    # process last axis
    strideargs = tuple([1 if i != len(sh)-1 else st[i] for i in range(len(sh))])
    x = stride(x, strideargs)
    shrinkargs = [(0,x.shape[i]) if i!=len(sh)-1 or x.shape[i]==sh[i] else (0,sh[i]) for i in range(len(sh))]
    x = shrink(x, tuple(shrinkargs))
    return x

# Setting logging level to INFO for demonstration purposes
logging.getLogger().setLevel(logging.WARNING)

# TODO check failing test cases and fix
if __name__ == '__main__':
    tests = construct_tests_nd(100)
    run_tests(tests, not_strided, offset=False)
    exit()
