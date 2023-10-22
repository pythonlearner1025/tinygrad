from tinygrad.tensor import Tensor
import torch
import numpy as np
import random

DEBUG=0

def pad_col_staggered(vector, pad_width, iaxis, kwargs):
    # Extract the number of elements to pad before and after the current axis
    A, i, colstoadd, axis = kwargs['A'],kwargs['i'],kwargs['colstoadd'],kwargs['axis']
    if axis != iaxis: return
    kwargs['i'] += 1
    _, pad_after = pad_width
    if colstoadd <= A.shape[axis]:
        # TODO: indexing logic should be general for n-dim
        vector[-pad_after:]= A[i][:colstoadd] if i<len(A) else 0
    else:
        #print('colstoadd logic')
        # colstoadd logic
        toadd,added,j = [],0,i
        while added < colstoadd:
            if j>=A.shape[axis-1]:
                toadd.extend([0]*(colstoadd-added))
                break
            # TODO: indexing logic should be general for n-dim
        #    print(j, colstoadd-added, A[j])
            toadd.extend(A[j][:colstoadd-added])
            added += len(A[j])
            j+=1
        #print(vector)
        vector[-pad_after:] = toadd 
    if DEBUG >= 1:
        print(f'i {i}')
        print(vector)

''' MovementOps(Enum): RESHAPE; PERMUTE; EXPAND; PAD; SHRINK; STRIDE;'''
# TODO 
# general to n-dim
# add offset
def not_strided_2d(x, sh, st, off):
    a,b = x.shape
    rowstoadd = 0
    while 1:
        if b*(a+rowstoadd)%st[0]==0: break
        rowstoadd+=1
    # pad cols 
    # this constructs A with rows with correct st[0]
    # this assumes st[1] is 1. If st[1] > 1, must pad each row w repeat values to get correct st[1]
    padargs, kwargs = ((0,rowstoadd),(0,0)), dict(mode='constant', constant_values=0)
    x = pad(x, padargs, **kwargs)
    # reshape
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

def run_tests(tests, offset=True):
    failed, faulty_tests = [], []
    t = 0
    for i,test in enumerate(tests):
        print(f'** TEST {i} **')
        x,sh,st_,o = test
        off = 0 if offset else o
        for i in st_[0]:
            for j in st_[1]:
                st = (i,j)
                flip = 0
                if i < j:
                    st, flip = (j,i), 1
                try:
                    a = torch.as_strided(x,sh,st,off)
                except:
                    faulty_tests.append((x,sh,st,off))
                    continue
                b = not_strided_2d(x, sh, st, off)
                if flip:
                    # NOTE remember to flip
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
        print(f'x {x.shape} sh {sh} st {st} off {off}')
        print()
        print(target)
        print(pred)
        print()
        o = st[1]-1
        colstoadd = (st[1]*sh[1]-o)-st[0]
        print(colstoadd)
    if DEBUG >= 1:
        print('** ALL FAULTY TESTS **')
        for fault in faulty_tests:
            x,sh,st,off = fault
            print(f'x {x.shape}, sh {sh} st {st} off {off}')
    print(f'** {len(failed)} FAILED {t-len(failed)} PASSED {100*len(failed)/t:.2f}% FAIL RATE')
if __name__ == '__main__':
    tests = construct_tests(100)
    run_tests(tests, offset=False)