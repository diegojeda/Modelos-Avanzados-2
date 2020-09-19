
# coding: utf-8

# In[ ]:


import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def import_imagenes():
   
    FF = h5py.File('data_F','r')
    
    CTF = FF.get('Frailejon')
    fraile = np.array(CTF)

    CTNF = FF.get('NoFrailejon')
    nofraile = np.array(CTNF)

    # Tomamos el numero de ejemplos para cada clase
    n = fraile.shape[0]
    m = nofraile.shape[0]

    # Inicializamos los arreglos donde vamos a guardar los ejemplos de cada clase
    fraile2 = []
    nofraile2 = []

    r = m

    for l in range(2):
        if(l==1):
            r = n
        for i in range(0,r,1):
            for j in range(0,5,1):
                if (j==0):
                    r1=0
                    c1=0
                    r2=70
                    c2=70
                    if(l==0):
                        x = nofraile[i,r1:r2,c1:c2,]
                        nofraile2.append(x)
                    if(l==1):
                        x = fraile[i,r1:r2,c1:c2,]
                        fraile2.append(x)
                if (j==1):
                    r1=r1+30
                    r2=r2+30
                    if(l==0):
                        x = nofraile[i,r1:r2,c1:c2,]
                        nofraile2.append(x)
                    if(l==1):
                        x = fraile[i,r1:r2,c1:c2,]
                        fraile2.append(x)
                if(j==2):
                    c1=c1+30
                    c2=c2+30
                    if(l==0):
                        x = nofraile[i,r1:r2,c1:c2,]
                        nofraile2.append(x)
                    if(l==1):
                        x = fraile[i,r1:r2,c1:c2,]
                        fraile2.append(x)
                if(j==3):
                    r1=0
                    r2=70
                    if(l==0):
                        x = nofraile[i,r1:r2,c1:c2,]
                        nofraile2.append(x)
                    if(l==1):
                        x = fraile[i,r1:r2,c1:c2,]
                        fraile2.append(x)
                if(j==4):
                    r1=15
                    c1=15
                    r2=85
                    c2=85
                    if(l==0):
                        x = nofraile[i,r1:r2,c1:c2,]
                        nofraile2.append(x)
                    if(l==1):
                        x = fraile[i,r1:r2,c1:c2,]
                        fraile2.append(x)

    nofraile2=np.asarray(nofraile2)
    fraile2=np.asarray(fraile2)

    CT_x2=np.concatenate((fraile2,nofraile2))

    # Aplanamos las imagenes
    CT_x_columna = CT_x2.reshape(CT_x2.shape[0], -1).T

    # Normalizamos los datos
    CT_xn = CT_x_columna/255.

    CT_y = ([1]*np.array(fraile2).shape[0]+[0]*np.array(nofraile2).shape[0])
    CT_y = np.vstack( CT_y ).T
        
    return CT_xn, CT_y

def particion_CE_CV(X,Y):

    CE_x, CV_x, CE_y, CV_y = train_test_split(X.T, Y.T, test_size = 0.3, random_state = 100)

    CE_x = CE_x.T
    CV_x = CV_x.T
    CE_y = CE_y.T
    CV_y = CV_y.T

    #n=CE_y.shape[1]
    #m=CV_y.shape[1]

    #print(CE_x.shape, CV_x.shape, CE_y.shape, CV_y.shape, n, m)
    return CE_x, CV_x, CE_y, CV_y

def print_errores(clases, X, y, p):

    a = p + y
    mis_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (20.0, 20.0) 
    num_ims = len(mis_indices[0])
    for i in range(num_ims):
        index = mis_indices[1][i]
        
        plt.subplot(2, num_ims, i + 1)
        plt.imshow(X[:,index].reshape(70,70,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Predicción: " + clases[int(p[0,index])] + " \n Clase: " + clases[y[0,index]])
