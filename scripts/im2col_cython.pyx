# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp

ctypedef fused floating:
    cnp.float32_t
    cnp.float64_t


def im2col_cython(cnp.ndarray[floating, ndim=4] x, int hh, int ww, int pad, int stride):
    cdef int n=x.shape[0], c=x.shape[1], h=x.shape[2], w=x.shape[3]
    cdef int oh=1+(h+2*pad-hh)//stride, ow=1+(w+2*pad-ww)//stride
    cdef cnp.ndarray[floating, ndim=4] xp=np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    cdef cnp.ndarray[floating, ndim=2] out=np.empty((c*hh*ww,n*oh*ow),dtype=x.dtype)
    cdef int ni,ci,ki,kj,yi,xi,row,col
    for ci in range(c):
        for ki in range(hh):
            for kj in range(ww):
                row=(ci*hh+ki)*ww+kj
                for ni in range(n):
                    for yi in range(oh):
                        for xi in range(ow):
                            col=(ni*oh+yi)*ow+xi
                            out[row,col]=xp[ni,ci,yi*stride+ki,xi*stride+kj]
    return out


def col2im_cython(cnp.ndarray[floating, ndim=2] cols, int n, int c, int h, int w,
                  int hh, int ww, int pad, int stride):
    cdef int oh=1+(h+2*pad-hh)//stride, ow=1+(w+2*pad-ww)//stride
    cdef cnp.ndarray[floating, ndim=4] out=np.zeros((n,c,h+2*pad,w+2*pad),dtype=cols.dtype)
    cdef int ni,ci,ki,kj,yi,xi,row,col
    for ci in range(c):
        for ki in range(hh):
            for kj in range(ww):
                row=(ci*hh+ki)*ww+kj
                for ni in range(n):
                    for yi in range(oh):
                        for xi in range(ow):
                            col=(ni*oh+yi)*ow+xi
                            out[ni,ci,yi*stride+ki,xi*stride+kj]+=cols[row,col]
    if pad == 0:
        return out
    return out[:,:,pad:pad+h,pad:pad+w]
