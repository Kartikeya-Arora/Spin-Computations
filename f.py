#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[6]:


import numpy as np

def func1(h,o):
    l=list()
    v,w=np.linalg.eigh(h)
    psi=np.array(w[:,0]) 
    
    hpsi=psi.conj().T
    
    for i in range(len(o)):
        x=np.dot(hpsi,np.dot(o[i],psi))
        l.append(x)
    return(l)  


# In[7]:



def func2(P,H,O,T):
 
    a={}
    l=list()
    HP=np.conj(P).T      
    nor=np.dot(HP,P)
    P=P/(nor**0.5)
    HP=HP/(nor**0.5)
    
    v,w=np.linalg.eigh(H)
    winv=np.transpose(np.conj(w))
    D=(np.diag(v))

    for i in range(len(O)):
        b = []
        
        for t in T:
            D=np.diag(np.exp((((-v)*t))*1.j))
            U=np.dot(w,np.dot(D,winv))
            psitR=np.dot(U,P)
            l.append(psitR)
            
            hpsitR=psitR.conj().T
            x=np.dot(hpsitR,np.dot(O[i],psitR))
            b.append(x[0])
            a[i]=b
            
            
        
    return(l,a) 


# In[ ]:




























































