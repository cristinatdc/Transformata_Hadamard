# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:44:39 2020

@author: CRIS
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

"""calculam in mod recursiv matricae Hadamard sub forma nenormalizata"""
def Hadamard(N):
    if N>2:
        Matrice1=np.concatenate((Hadamard(N/2),Hadamard(N/2)),axis=1,out=None)
        Matrice2=np.concatenate((Hadamard(N/2),-Hadamard(N/2)),axis=1,out=None)
        Matrice3=np.concatenate((Matrice1,Matrice2),axis=0,out=None)
        H=Matrice3
    else:
        aux2=1
        A=[[1,1],[1,-1]]
        H=np.dot(aux2,A)
    return H

"""functie ce ordoneaza o matrice primita ca si parametru
in functie de numarul de schimbari de semn de pe fiecare linie"""
def ordonare(H):
    nr_linii=len(H)
    nr_coloane=len(H[0])
    v=np.zeros(nr_linii,dtype=int)
    k=0
    for i in range(nr_linii):
        nr=0;
        for j in range(nr_coloane-1):
            if H[i][j] != H[i][j+1]:
                nr+=1
        v[k]=nr
        k=k+1
    #print(v)
    """ in vectorul am obtinut valorea numarului schimbarilor de semn pentru fiecare linie in parte, 
    in ordinea liniilor
    """ 
    H_nou=np.zeros((nr_linii,nr_coloane))
    i=0
    for k in range(nr_linii):
        for j in range(nr_coloane):
            H_nou[v[k]][j]=H[i][j]
        i+=1
    return H_nou

"""suma patratelor modulelor coeficientilor"""
def energie(M,N):
    suma=0
    for m in range(0,N):
        for n in range(0,N):
            suma+=math.pow(np.abs(M[m][n]),2)
    return suma
"""functia main"""
if __name__ == "__main__":
    """Generarea matricii Hadamard"""
    print("Test functionare algoritm: ")
    Nn=8 #ordinul matricii
    H8=Hadamard(Nn) #apelul functiei
    print("\nMatricea Hadamard de ordinul 8: ")
    print(H8)
    H8_ord=ordonare(H8) #apelul functiei care returneaza matricea ordonata
    print("\nMatricea Hadamard de ordinul 8 ordonata: ")
    print(H8_ord)

    """Import imagine de test"""
    Imagine_de_intrare="PeppersGri.bmp"
    U=cv2.imread(Imagine_de_intrare,0).astype(float)

    print("\nImaginea originala: \n")
    plt.imshow(U,cmap='gray')
    plt.show()

    print("\nHistograma imaginii originale: ")
    plt.figure()
    (n,bins,patches)=plt.hist(U.ravel(),256,[0,255],density=True,cumulative=True)
    plt.show()

    print("\nMatricea imagine: ")
    print(U,"\n")
    nr_linii=len(U)
    print("\nNumarul de linii: ",nr_linii)
    nr_coloane=len(U[0])
    print("\nNumarul de coloane: ",nr_coloane)

    if nr_linii==nr_coloane:
        """Generarea matricii Hadamard"""
        N=nr_linii
        H=Hadamard(N)
        
        """Normalizare"""
        var=1/math.sqrt(N)
        H=np.dot(var,H)
        print("\nMatricea Hadamard de ordinul ",N,":")
        print(H,"\n")
        H_ord=ordonare(H)
        print("\nMatricea Hadamard de ordinul ",N," ordonata: ")
        print(H_ord,"\n")

        """Transformata Hadamard
        V=H*U*H
        """
        aux=np.dot(H,U) #H*U
        V=np.dot(aux,H) #(H*U)*H
        
        print("\nMatricea transformarii in forma neordonata: ")
        print(V)
        print("\nImaginea transformata(in forma neordonata): ")
        plt.figure()
        plt.imshow(V.astype(np.uint8),cmap = 'gray')
        plt.show()
             
        print("\nHistograma imaginii in domeniul frecventa: \n")
        plt.figure()
        (n,bins,patches)=plt.hist(V.ravel(),256,[0,255],density=True,cumulative=True)
        plt.show()
        
        aux=np.dot(H_ord,U)
        V2=np.dot(aux,H_ord)
        print("\nMatricea transformarii in forma ordonata: ")
        print(V2,"\n")

        print("\nImaginea transformata(in forma ordonata): ")
        plt.figure()
        plt.imshow(V2.astype(np.uint8),cmap = 'gray')
        plt.show()

        print("\nHistograma imaginii in domeniul frecventa: ")
        plt.figure()
        (n,bins,patches)=plt.hist(V2.ravel(),256,[0,255],density=True,cumulative=True)
        plt.show()
         
        """Conservarea energiei"""
        membru_stang=energie(V,N)
        print("\nEnergia in domeniul imaginii: ",membru_stang,"\n")
        membru_drept=energie(U,N)
        print("\nEnergia in domeniul transformat: ",membru_drept,"\n")
        if membru_stang==membru_drept:
            print("\nENERGIA SE CONSERVA!!!")
        else:
            print("\nENERGIA NU SE CONSERVA!!!")
      
        """Imaginea refacuta"""
        aux=np.dot(np.transpose(np.linalg.inv(H_ord)),V2)
        V_out=np.dot(aux,np.transpose(H_ord))
        print("\nImaginea refacuta: ")
        plt.figure()
        plt.imshow(V_out,cmap='gray')
        plt.show()
        print("\nHistograma imaginii refacute")
        plt.figure()
        (n,bins,patches)=plt.hist(V_out.ravel(),256,[0,255],density=True,cumulative=True)
        plt.show()
        
        """Eroarea la refacere"""
        print('\nEroarea la refacere: ')
        mse = (np.square(U.astype(float) - V_out.astype(float))).mean()
        print('MSE = ',mse)

        """Filtrare in domeniul frecventa"""
        PercentKeptCoeffs=0.2
        H,W=V2.shape
        CoeffsMask=np.zeros(V2.shape)
        CoeffsMask[:np.int(PercentKeptCoeffs*H),:np.int(PercentKeptCoeffs*W)]=1
        print("\nMasca de filtrare: ")
        plt.figure()
        plt.imshow((CoeffsMask*255).astype(np.uint8),cmap = 'gray')
        plt.show()
        print("\nMasca de filtrare sub forma matriceala: ")
        print(CoeffsMask)
        
        """Se inmulteste matricea imaginii in domeniul transformat cu masca de filtrare definita anterior"""   
        FiltImg=np.multiply(V2,CoeffsMask)
        print("\nMatricea dupa filtrare: ")
        print(FiltImg)
        print("\nImaginea in domeniul frecventa: ")
        plt.figure()
        plt.imshow(FiltImg,cmap='gray')
        plt.show()
        
        """Compactarea energiei"""
        nr=0
        for i in range(nr_linii):
            for j in range(nr_coloane):
                if np.abs(FiltImg[i][j])!=0:
                    nr+=1
        p=nr/(nr_linii*nr_coloane)
        print("\nCoeficientul de compactare este de",p*100,"%. ")
        
        """Aplicam transfromata inversa"""
        X=np.linalg.inv(H_ord)
        OutImgFilt=(np.dot(np.dot(X,FiltImg),H_ord)).astype(np.uint8)
        
        print("\nRezultatul filtrarii: ")
        plt.figure()
        plt.imshow(OutImgFilt,cmap='gray')
        plt.show()
        print("\nHistograma imaginii filtrate: ")
        plt.figure()
        (n,bins,patches)=plt.hist(OutImgFilt.ravel(),256,[0,255],density=True,cumulative=True)
        plt.show()
               
        """Eroarea la refacere"""
        print('\nComparam imaginea filtrata cu imaginea originala.')
        mse = (np.square(U.astype(float) - OutImgFilt.astype(float))).mean()
        print('MSE = ',mse)