#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:27:14 2020

@author: whadymacbook2016
"""

#ejemplo differential quadrature RBF para Pennes equation CV RBF
#implicito descentrado en t solo collocation CON cvrbf en la parte de arriba
#con todos los cambios y mejoras del porgrama de lag 




import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse.linalg import factorized,gmres
from scipy.sparse import csc_matrix


rho=1000
k=0.628
cp=4187
rhob=1.06e3
cpb=3860
wb=1.87e-3
Tb=37
Qm=1.19e3
To=Tb
Ro=0
rp=1e-2
dt=0.01
t_heat=1
t_final=5
xn=[]
L=5.0e-2
H=5.0e-2
nl=120  #number of nodes in each direction
nh=120
dx=L/(nl-1)
dy=H/(nh-1)
c=3*np.sqrt(dx**2+dy**2) #rbf parameter
ntot=nl*nh
nsteps=int(t_final/dt)
MT=np.zeros((ntot,nsteps+1))
xl=np.linspace(0,L,nl)
yh=np.linspace(0,H,nh)
for xj in yh:
    for xi in xl:
        xn.append([xi,xj])

def qp(r,t):
    if r<=rp and t<=t_heat and t>0:
        q=-4.0*(100**2)
    else:
        q=0
    return q    
   
def Qsource(t):
    if t<=t_heat and t>0:
        q=0
    else:
        q=0
    return q    

#rbf and its derivatives
def f(x,xs):
  r=np.sqrt(np.dot(x-xs,x-xs))  
  #multiquadrics
  y=np.sqrt(r**2+c**2)  
  #thin plate spline
  # if r!=0:
  #   y=r**2*np.log(r)
  # else:
  #   y=0  
  return y

def df(x,xs): #dy[0]=0 x derivative dy[1] y derivative
  dy=np.zeros(2)
  r=np.sqrt(np.dot(x-xs,x-xs))  
  #multiquadrics
  dy[0]=(x[0]-xs[0])/f(x,xs)
  dy[1]=(x[1]-xs[1])/f(x,xs)
  #thin plate spline
  # if r!=0:
  #   dy=(x-xs)+2*(x-xs)*np.log(r)
  return dy

def d2f(x,xs): #secodn derivative matrix [[Dxx Dxy][Dyx Dyy]]
  d2y=np.zeros((2,2))
  r=np.sqrt(np.dot(x-xs,x-xs))  
  #multiquadrics
  d2y[0,0]=-(x[0]-xs[0])**2/f(x,xs)**3+1/f(x,xs)
  d2y[0,1]=-(x[0]-xs[0])*(x[1]-xs[1])/f(x,xs)**3
  d2y[1,0]=d2y[0,1]
  d2y[1,1]=-(x[1]-xs[1])**2/f(x,xs)**3+1/f(x,xs)    
  return d2y

#calculos en los stencils con DQ
# def dfsten(x,xs):
#     n=np.size(xs,0)
#     F=np.zeros((n,n))
#     Bx=np.zeros(n)
#     By=np.zeros(n)
#     wx=np.zeros(n)
#     wy=np.zeros(n)
#     for i in range(n):
#         Bx[i]=df(x,xs[i,:])[0]
#         By[i]=df(x,xs[i,:])[1]
#         for j in range(n):
#             F[i,j]=f(xs[j,:],xs[i,:])
#     wx=np.linalg.solve(F,Bx)
#     wy=np.linalg.solve(F,By)
#     return wx,wy

# def d2fsten(x,xs):
#     n=np.size(xs,0)
#     F=np.zeros((n,n))
#     Bxx=np.zeros(n)
#     Bxy=np.zeros(n)
#     Byy=np.zeros(n)
#     wxx=np.zeros(n)
#     wxy=np.zeros(n)
#     wyy=np.zeros(n)
#     for i in range(n):
#         Bxx[i]=d2f(x,xs[i,:])[0,0]
#         Bxy[i]=d2f(x,xs[i,:])[0,1]
#         Byy[i]=d2f(x,xs[i,:])[1,1]
#         for j in range(n):
#             F[i,j]=f(xs[j,:],xs[i,:])
#     wxx=np.linalg.solve(F,Bxx)
#     wxy=np.linalg.solve(F,Bxy)
#     wyy=np.linalg.solve(F,Byy)
#     return wxx,wxy,wyy

#calculos en los stencils con local rbf normal NO DQ
def dfsten(x,xs):
    n=np.size(xs,0)
    F=np.zeros((n+1,n+1))
    Finv=np.zeros((n+1,n+1))
    Bx=np.zeros(n+1)
    By=np.zeros(n+1)
    wx=np.zeros(n)
    wy=np.zeros(n)
    for i in range(n):
        for j in range(n):
            F[i,j]=f(xs[i,:],xs[j,:])
    F[n,0:n]=1  
    F[0:n,n]=1
    Finv=np.linalg.inv(F)
    for i in range(n):
        Bx[i]=df(x,xs[i,:])[0]
        By[i]=df(x,xs[i,:])[1]
    wx=np.matmul(Bx,Finv)[0:n]
    wy=np.matmul(By,Finv)[0:n]
    return wx,wy

def d2fsten(x,xs):
    n=np.size(xs,0)
    F=np.zeros((n+1,n+1))
    Finv=np.zeros((n+1,n+1))
    Bxx=np.zeros(n+1)
    Bxy=np.zeros(n+1)
    Byy=np.zeros(n+1)
    wxx=np.zeros(n)
    wxy=np.zeros(n)
    wyy=np.zeros(n)
    for i in range(n):
        for j in range(n):
            F[i,j]=f(xs[i,:],xs[j,:])
    F[n,0:n]=1  
    F[0:n,n]=1
    Finv=np.linalg.inv(F)
    for i in range(n):
        Bxx[i]=d2f(x,xs[i,:])[0,0]
        Bxy[i]=d2f(x,xs[i,:])[0,1]
        Byy[i]=d2f(x,xs[i,:])[1,1]
    wxx=np.matmul(Bxx,Finv)[0:n]
    wxy=np.matmul(Bxy,Finv)[0:n]
    wyy=np.matmul(Byy,Finv)[0:n]
    return wxx,wxy,wyy    
        

#definicion de familias conectividades
f1=[] #inferior
f2=[] #derecha
f3=[] #superior       
f4=[] #izquierda       
f5=[] #esquina inf izq       
f6=[] #esq inf der       
f7=[] #esq sup der       
f8=[] #es sup izq       
f9=[] #internos

for n in xn:
  if n[0]>0 and n[0]<L and n[1]==0:
    f1.append(xn.index(n))       
  if n[0]==L and n[1]>0 and n[1]<H:
    f2.append(xn.index(n))       
  if n[0]>0 and n[0]<L and n[1]==H:
    f3.append(xn.index(n))       
  if n[0]==0 and n[1]>0 and n[1]<H:
    f4.append(xn.index(n))       
  if n[0]==0 and n[1]==0:
    f5.append(xn.index(n))       
  if n[0]==L and n[1]==0:
    f6.append(xn.index(n))       
  if n[0]==L and n[1]==H:
    f7.append(xn.index(n))       
  if n[0]==0 and n[1]==H:
    f8.append(xn.index(n))       
  if n[0]>0 and n[0]<L and n[1]>0 and n[1]<H:
    f9.append(xn.index(n))       
   
       

#Matrix assembly
A=np.zeros((ntot,ntot))
B=np.zeros(ntot)
Tj2=np.zeros(ntot)
Tj1=np.zeros(ntot)
Tj0=np.zeros(ntot)

Tj0[:]=To
MT[:,0]=To
 

#A matrix assembly
dst_f1=[]   #para general los estencils una sola vez
st_f1=[]
for i in f1:#inferior
    x=np.array(xn[i])
    st=[i-1,i,i+1,i-1+nl,i+nl,i+nl+1,i+2*nl-1,i+2*nl,i+2*nl+1]
    st_f1.append(st)
    xs=np.array([xn[m] for m in st])
    dz=dfsten(x,xs)
    dst_f1.append(dz)
    for m in range(np.size(st)):
        A[i,st[m]]=k*dz[1][m]
dst_f2=[]
st_f2=[]
for i in f2:#derecha
    x=np.array(xn[i])
    st=[i-nl-2,i-nl-1,i-nl,i-2,i-1,i,i+nl-2,i+nl-1,i+nl]
    st_f2.append(st)
    xs=np.array([xn[m] for m in st])
    dr=dfsten(x,xs)
    dst_f2.append(dr)
    for m in range(np.size(st)):
        A[i,st[m]]=-k*dr[0][m]
dst_f3=[]
st_f3=[]
dzsouth_f3=[]
dznorth_f3=[]
dzwest_f3=[]
dzeast_f3=[]
for i in f3:#superior
    x=np.array(xn[i])
    st=[i-1,i,i+1,i-nl-1,i-nl,i-nl+1,i-2*nl-1,i-2*nl,i-2*nl+1]
    st_f3.append(st)
    xs=np.array([xn[m] for m in st])
    dz=dfsten(x,xs)
    dst_f3.append(dz)
    xsouth=x+np.array([0,-dy/2])
    xwest=x+np.array([-dx/2,-dy/4])
    xeast=x+np.array([dx/2,-dy/4])
    dzsouth=dfsten(xsouth,xs)[1]
    dznorth=dz[1]    
    dzwest=dfsten(xwest,xs)[0]
    dzeast=dfsten(xeast,xs)[0]
    dzsouth_f3.append(dzsouth)
    dznorth_f3.append(dznorth)
    dzwest_f3.append(dzwest)
    dzeast_f3.append(dzeast)
    Awest=2*np.pi*xwest[0]*dy/2
    Aeast=2*np.pi*xeast[0]*dy/2
    Asouth=np.pi*(xeast[0]**2-xwest[0]**2)
    Anorth=Asouth
    dv=Asouth*dy/2
    for m in range(np.size(st)):
        A[i,st[m]]=k*Asouth*dzsouth[m]\
        +k*Awest*dzwest[m]-k*Aeast*dzeast[m]
    A[i,i]+=(rho*cp)/dt*dv+wb*rhob*cpb*dv
dst_f4=[]
st_f4=[]
for i in f4:#izquierda
    x=np.array(xn[i])
    st=[i-nl,i-nl+1,i-nl+2,i,i+1,i+2,i+nl,i+nl+1,i+nl+2]
    st_f4.append(st)
    xs=np.array([xn[m] for m in st])
    dr=dfsten(x,xs)
    dst_f4.append(dr)
    for m in range(np.size(st)):
        A[i,st[m]]=k*dr[0][m]
dst_f5=[]
st_f5=[]
for i in f5:#esq inferior izquierda
    x=np.array(xn[i])
    st=[i,i+1,i+2,i+nl,i+nl+1,i+nl+2,i+2*nl,i+2*nl+1,i+2*nl+2]
    st_f5.append(st)
    xs=np.array([xn[m] for m in st])
    dz=dfsten(x,xs)
    dst_f5.append(dz)
    for m in range(np.size(st)):
        A[i,st[m]]=k*dz[1][m]
dst_f6=[]
st_f6=[]
for i in f6:#esq inferior derecha
    x=np.array(xn[i])
    st=[i-2,i-1,i,i-2+nl,i-1+nl,i+nl,i-2+2*nl,i-1+2*nl,i+2*nl]
    st_f6.append(st)
    xs=np.array([xn[m] for m in st])
    dz=dfsten(x,xs)
    dst_f6.append(dz)
    for m in range(np.size(st)):
        A[i,st[m]]=k*dz[1][m]
dst_f7=[]
st_f7=[]
dzsouth_f7=[]
dznorth_f7=[]
dzwest_f7=[]
dzeast_f7=[]
for i in f7:#esq sup derecha
    x=np.array(xn[i])
    st=[i-2-2*nl,i-1-2*nl,i-2*nl,i-2-nl,i-1-nl,i-nl,i-2,i-1,i]
    st_f7.append(st)
    xs=np.array([xn[m] for m in st])
    dz=dfsten(x,xs)
    dst_f7.append(dz)
    xsouth=x+np.array([-dx/4,-dy/2])
    xwest=x+np.array([-dx/2,-dy/4])
    xeast=x+np.array([0,-dy/4])
    dzsouth=dfsten(xsouth,xs)[1]
    dznorth=dz[1]
    dzwest=dfsten(xwest,xs)[0]
    dzeast=dfsten(xeast,xs)[0]
    dzsouth_f7.append(dzsouth)
    dznorth_f7.append(dznorth)
    dzwest_f7.append(dzwest)
    dzeast_f7.append(dzeast)
    Awest=2*np.pi*xwest[0]*dy/2
    Aeast=2*np.pi*xeast[0]*dy/2
    Asouth=np.pi*(xeast[0]**2-xwest[0]**2)
    Anorth=Asouth
    dv=Asouth*dy/2
    for m in range(np.size(st)):
        A[i,st[m]]=k*Asouth*dzsouth[m]+k*Awest*dzwest[m]
    A[i,i]+=(rho*cp)/dt*dv+wb*rhob*cpb*dv
dst_f8=[]
st_f8=[]
dzsouth_f8=[]
dznorth_f8=[]
dzwest_f8=[]
dzeast_f8=[]
for i in f8:#esq sup izq
    x=np.array(xn[i])
    st=[i-2*nl,i-2*nl+1,i-2*nl+2,i-nl,i-nl+1,i-nl+2,i,i+1,i+2]
    st_f8.append(st)
    xs=np.array([xn[m] for m in st])
    dz=dfsten(x,xs)
    dst_f8.append(dz)
    xsouth=x+np.array([dx/4,-dy/2])
    xwest=x+np.array([0,-dy/4])
    xeast=x+np.array([dx/2,-dy/4])
    dzsouth=dfsten(xsouth,xs)[1]
    dznorth=dz[1]
    dzwest=dfsten(xwest,xs)[0]
    dzeast=dfsten(xeast,xs)[0]
    dzsouth_f8.append(dzsouth)
    dznorth_f8.append(dznorth)
    dzwest_f8.append(dzwest)
    dzeast_f8.append(dzeast)
    Awest=2*np.pi*xwest[0]*dy/2
    Aeast=2*np.pi*xeast[0]*dy/2
    Asouth=np.pi*(xeast[0]**2-xwest[0]**2)
    Anorth=Asouth
    dv=Asouth*dy/2
    for m in range(np.size(st)):
        A[i,st[m]]=k*Asouth*dzsouth[m]-k*Aeast*dzeast[m]
    A[i,i]+=(rho*cp)/dt*dv+wb*rhob*cpb*dv
dst_f9=[]
st_f9=[]
for i in f9:#internos
    x=np.array(xn[i])
    st=[i-nl-1,i-nl,i-nl+1,i-1,i,i+1,i+nl-1,i+nl,i+nl+1]
    st_f9.append(st)
    xs=np.array([xn[m] for m in st])
    lap=d2fsten(x,xs)[0]+1/x[0]*dfsten(x,xs)[0]+d2fsten(x,xs)[2]
    dst_f9.append(lap)
    for m in range(np.size(st)):
        A[i,st[m]]=-k*lap[m]
    A[i,i]+=(rho*cp)/dt+wb*rhob*cpb

#ensamblar bloques
#A1=np.concatenate((ART,ARR),axis=1)
#A2=np.concatenate((ATT,ATR),axis=1)
#LU factor of A      
#A_lu, A_piv = lu_factor(A) 
A_csc=csc_matrix(A) 
solve=factorized(A_csc)  

time=np.zeros(nsteps+1)
for j in range(1,nsteps+1):
#RHS vector B assembly
    t=j*dt
    time[j]=t
    for i in f1:#inferior
        x=np.array(xn[i])
        st=st_f1[f1.index(i)]
        xs=np.array([xn[m] for m in st])
#        dz=dfsten(x,xs)
        dz=dst_f1[f1.index(i)]
        T0st=np.array([Tj0[m] for m in st])
        T1st=np.array([Tj1[m] for m in st])
        B[i]=0
    for i in f2:#derecha
        x=np.array(xn[i])
        st=st_f2[f2.index(i)]
        xs=np.array([xn[m] for m in st])
#        dr=dfsten(x,xs)
        dr=dst_f2[f2.index(i)]
        T0st=np.array([Tj0[m] for m in st])
        T1st=np.array([Tj1[m] for m in st])        
        B[i]=0
    for i in f3:#superior
        x=np.array(xn[i])
        st=st_f3[f3.index(i)]
        xs=np.array([xn[m] for m in st])
        dz=dst_f3[f3.index(i)]
        dzsouth=dzsouth_f3[f3.index(i)]
        dznorth=dznorth_f3[f3.index(i)]    
        dzwest=dzwest_f3[f3.index(i)] 
        dzeast=dzeast_f3[f3.index(i)] 
        xsouth=x+np.array([0,-dy/2])
        xwest=x+np.array([-dx/2,-dy/4])
        xeast=x+np.array([dx/2,-dy/4])
        Awest=2*np.pi*xwest[0]*dy/2
        Aeast=2*np.pi*xeast[0]*dy/2
        Asouth=np.pi*(xeast[0]**2-xwest[0]**2)
        Anorth=Asouth
        dv=Asouth*dy/2
        T0st=np.array([Tj0[m] for m in st] )
        T1st=np.array([Tj1[m] for m in st])        
        B[i]=(wb*rhob*cpb*Tb+(rho*cp)/dt*Tj0[i]+Qm+Qsource(t))*dv\
        -qp(x[0],t)*Anorth
    for i in f4:#izquierda
        x=np.array(xn[i])
        st=st_f4[f4.index(i)]
        xs=np.array([xn[m] for m in st])
#        dr=dfsten(x,xs)
        dr=dst_f4[f4.index(i)]
        T0st=np.array([Tj0[m] for m in st])
        T1st=np.array([Tj1[m] for m in st])        
        B[i]=0
    for i in f5:#esq inferior izquierda
        x=np.array(xn[i])
        st=st_f5[f5.index(i)]
        xs=np.array([xn[m] for m in st])
#        dz=dfsten(x,xs)
        dz=dst_f5[f5.index(i)]
        T0st=np.array([Tj0[m] for m in st])
        T1st=np.array([Tj1[m] for m in st])        
        B[i]=0
    for i in f6:#esq inferior derecha
        x=np.array(xn[i])
        st=st_f6[f6.index(i)]
        xs=np.array([xn[m] for m in st])
#        dz=dfsten(x,xs)
        dz=dst_f6[f6.index(i)]
        T0st=np.array([Tj0[m] for m in st])
        T1st=np.array([Tj1[m] for m in st])        
        B[i]=0
    for i in f7:#esq sup derecha
        x=np.array(xn[i])
        st=st_f7[f7.index(i)]
        xs=np.array([xn[m] for m in st])
#        dz=dfsten(x,xs)
        dz=dst_f7[f7.index(i)]
        xsouth=x+np.array([-dx/4,-dy/2])
        xwest=x+np.array([-dx/2,-dy/4])
        xeast=x+np.array([0,-dy/4])
        dzsouth=dzsouth_f7[f7.index(i)]
        dznorth=dznorth_f7[f7.index(i)]    
        dzwest=dzwest_f7[f7.index(i)] 
        dzeast=dzeast_f7[f7.index(i)] 
        Awest=2*np.pi*xwest[0]*dy/2
        Aeast=2*np.pi*xeast[0]*dy/2
        Asouth=np.pi*(xeast[0]**2-xwest[0]**2)
        Anorth=Asouth
        dv=Asouth*dy/2
        T0st=np.array([Tj0[m] for m in st])
        T1st=np.array([Tj1[m] for m in st])        
        B[i]=(wb*rhob*cpb*Tb+(rho*cp)/dt*Tj0[i]+Qm+Qsource(t))*dv\
        -qp(x[0],t)*Anorth
    for i in f8:#esq sup izq
        x=np.array(xn[i])
        st=st_f8[f8.index(i)]
        xs=np.array([xn[m] for m in st])
#        dz=dfsten(x,xs)
        dz=dst_f8[f8.index(i)]
        xsouth=x+np.array([dx/4,-dy/2])
        xwest=x+np.array([0,-dy/4])
        xeast=x+np.array([dx/2,-dy/4])
        dzsouth=dzsouth_f8[f8.index(i)]
        dznorth=dznorth_f8[f8.index(i)]    
        dzwest=dzwest_f8[f8.index(i)] 
        dzeast=dzeast_f8[f8.index(i)] 
        Awest=2*np.pi*xwest[0]*dy/2
        Aeast=2*np.pi*xeast[0]*dy/2
        Asouth=np.pi*(xeast[0]**2-xwest[0]**2)
        Anorth=Asouth
        dv=Asouth*dy/2
        T0st=np.array([Tj0[m] for m in st])
        T1st=np.array([Tj1[m] for m in st])        
        B[i]=(wb*rhob*cpb*Tb+(rho*cp)/dt*Tj0[i]+Qm+Qsource(t))*dv\
        -qp(x[0],t)*Anorth
    for i in f9:#internos
        x=np.array(xn[i])
        st=st_f9[f9.index(i)]
        xs=np.array([xn[m] for m in st])
#        lap=d2fsten(x,xs)[0]+1/x[0]*dfsten(x,xs)[0]+d2fsten(x,xs)[2]
        lap=dst_f9[f9.index(i)]
        T0st=np.array([Tj0[m] for m in st])
        T1st=np.array([Tj1[m] for m in st])        
        B[i]=wb*rhob*cpb*Tb+(rho*cp)/dt*Tj0[i]+Qm+Qsource(t)    

#solucion        
#    Tj2 = lu_solve((A_lu, A_piv), B)
    Tj=solve(B)
#    Tj2=np.linalg.solve(A,B)
# #    Tj1=np.linalg.solve(A,B)
#tiem store and update
    MT[:,j]=Tj1
    Tj0=Tj1
    
#Tj1=np.linalg.solve(A,B)    
    #plt.figure()
#plt.spy(A)       

plt.figure()
plt.plot(time,MT[nl*nh-nl,:])

bioheat=np.loadtxt('pennes_detailed.txt')

plt.plot(bioheat[:,0],bioheat[:,1]) 


        
    


  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

     
        
   
       
       
        
        
        













