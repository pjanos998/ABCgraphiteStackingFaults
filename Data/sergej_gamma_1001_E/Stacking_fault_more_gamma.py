import sys
import numpy as np
from itertools import combinations
import regex as re
import os

#Elsőnek építsünk fel egy sima SSH-t, az a tuti.
a0 = 1; #length of unit cell vector.
a1 = a0/2*np.array([np.sqrt(3), 1, 0]);
a2 = a0/2*np.array([np.sqrt(3),-1, 0]);
a3 = a0/2*np.array([0 ,0, 1]);

#reciprocal lattice vectors:
b1 = 2*np.pi/a0*np.matrix([1/np.sqrt(3),1]);
b2 = 2*np.pi/a0*np.matrix([1/np.sqrt(3),-1]);
a_cc = a0/np.sqrt(3); #carbon-carbon distance

ossz_kombok = []
for N in range(0,11):
    faults = [f'{i}b' for i in range(1,N)]
    kombok = []
    for i in range(int(np.ceil(N/2))):
        komb = list(combinations(faults,i))
        kombok.append(komb[:int(np.ceil(len(komb)/2))])
    ossz_kombok.append(kombok)
    
ossz_kombok_unzipped = []
for i in range(0,len(ossz_kombok)):
    aktual_komb = []
    for j in range(0,len(ossz_kombok[i])):
        for k in range(len(ossz_kombok[i][j])):
            aktual_komb.append(list(ossz_kombok[i][j][k]))
    ossz_kombok_unzipped.append(aktual_komb)

def k_path(path_array,nk):
    dist=np.linalg.norm(np.diff(path_array,axis=0),axis=2)
    kx1=np.array([])
    ky1=np.array([])
    kstep1=np.array([0])
    for i in np.arange(0,np.shape(path_array)[0]-1):
        x1=path_array[i][0,0]
        y1=path_array[i][0,1]
        x2=path_array[i+1][0,0]
        y2=path_array[i+1][0,1]
        if x2==x1:
            kx=np.zeros(int(nk/dist.sum()*dist[i]))
            ky=np.linspace(y1,y2,int(nk/dist.sum()*dist[i]))+x1
        else:
            b=(x1*y2 - x2*y1)/(x1-x2)
            m=(y2-y1)/(x2-x1)
            kx=np.linspace(x1,x2,int(nk/dist.sum()*dist[i]))
            ky=kx*m+b
        kx1=np.concatenate((kx1,kx))
        ky1=np.concatenate((ky1,ky))
        kstep=dist[0]/int(nk/dist.sum()*dist[0])
        kstep1=np.concatenate((kstep1,kstep))
    return kx1,ky1

def hk(D,t0,t1,t3,t4,N,in_list,k):
    '''Nagyon hasonló a chain_maker kódhoz, csak itt már megadhatjuk, hogy milyen erősségű legyen a kötés. És meg kell adni egy listát összetartozó kx, ky párokkal!'''
    #Először kell az alap hamiltoni, azaz feltölteni SSH-val egy 2*N nagyságú alap mátrixot.
    kx=k[0]
    ky=k[1]
    F = 1 + np.exp(1j/2*(np.sqrt(3)*kx+ky)) + np.exp(1j/2*(np.sqrt(3)*kx-ky))
    F_A1B2 = np.exp(1j*ky*3) + np.exp(1j/2*(np.sqrt(3)*kx+ky)) + np.exp(1j/2*(np.sqrt(3)*kx-ky))
    #F_A1B2 = F
    
    h0_ABC = np.array([[0*np.ones(np.shape(F)),t0*F],
                       [np.conj(t0*F),0*np.ones(np.shape(F))]]).transpose((2,0,1))
    
    D_mat = np.array([[0*np.ones(np.shape(F)),0*np.ones(np.shape(F))],
                      [0*np.ones(np.shape(F)),D*np.ones(np.shape(F))]]).transpose((2,0,1))
    
    D_mat_b = np.array([[D*np.ones(np.shape(F)),0*np.ones(np.shape(F))],
                      [0*np.ones(np.shape(F)),0*np.ones(np.shape(F))]]).transpose((2,0,1))
    
    h1 = np.array([[t4*F,t3*F_A1B2],
                   [t1*np.ones(np.shape(F)),t4*F]]).transpose((2,0,1))
    
    h1_b = np.conj(np.array([[t4*F,t1*np.ones(np.shape(F))],
                             [t3*F_A1B2,t4*F]])).transpose((2,0,1))
    
    
    H0k = np.zeros((len(F),2*N,2*N),dtype=complex)
    
    onsite_mat = np.kron(np.eye(N),h0_ABC)
    #Ez a fenti rész nem változott semmit, ez a nagyon jó a helyesen megválasztott címkézésben.
    
    #Itt jön a gondolkodós rész, hogy pontosan hogyan kell megcímkézni az adott nagyságú mátrixok megfelelő pontjait. 
    #Mert azokra kell majd hivatkozni.
    
    #most jön az átírás, amikor van valami input hiba
    
    on_site_list = np.ones(N)
    on_site_list[-1] = 0
    on_site_list_b     = np.ones(N)
    on_site_list_b[0] = 0
    hoppingok_list = np.ones(N-1) 
    flag_T = False
    print(in_list)
    if (in_list == ['']):
        in_list = []
    for string in in_list:
        helyzet = (int(re.search(r'\d+', string).group())) #kiszedjük, hogy melyik rétegre gondolt a költő
        hoppingok_list[helyzet-1]   -= 1
        on_site_list[helyzet-1]   -= 1
        on_site_list[helyzet]     += 1
        on_site_list_b[helyzet-1] += 1
        on_site_list_b[helyzet]   -= 1
    
    hopping_mat = np.kron(np.diag(hoppingok_list,k=1),h1) + np.kron(np.diag(1-hoppingok_list,k=1),h1_b)
    hopping_mat += np.conj(np.transpose(hopping_mat,[0,2,1]))
    onsite_mat += np.kron(np.diag(on_site_list),D_mat) + np.kron(np.diag(on_site_list_b),D_mat_b)
    
    H0k += onsite_mat
    H0k += hopping_mat #hermitikus a mátrix, ezért ez kell ide is
    return H0k

def grid_maker(vec1,vec2,num,misplace=[0,0]):
    points = []
    for i in range(num):
        for j in range(num):
            points.append([misplace[0] + i*vec1[0]/num + j*vec2[0]/num,misplace[1] + i*vec1[1]/num + j*vec2[1]/num])
    return np.array(points).reshape(num*num,2)

def DOS(E,Energies,alpha,norm):
    Energies = Energies.flatten()
    D = norm/alpha * np.exp(-((Energies[:,None] - E)**2)/(2*alpha**2))
    dos = np.sum(D,axis=0)
    return dos

def PDOS(E,Energies,w,alpha,norm):
    Energies = Energies.flatten()
    w = np.transpose(w, axes=[0,2,1])
    w = abs(w.reshape(w.shape[0]*w.shape[1], w.shape[2]))**2
    D = norm/alpha * w.T @ np.exp(-((Energies[:,None] - E)**2)/(2*alpha**2))
    dos = np.sum(D,axis=0)
    return dos, D


def Bands_DOS(D, t0, t1, t3, t4, N, in_list, k1, k2, E, alpha, norm=1, norm_from=-3.5):
    Energy_d, vectors_d=(np.linalg.eigh(hk(D,t0,t1,t3,t4,N,in_list,k2)))
    d_normed, pdos_normed = PDOS(E,Energy_d,vectors_d,alpha,norm)

    return [d_normed, pdos_normed]



def main(D, t0, t1, t3, t4, N, in_list, E, E_num, grid_num, alpha):
    G=np.array(0*b1) #gamma pont az origo
    K=np.array((b1-b2)/3)
    M=np.array(b1/2)
    kx_band,ky_band=k_path(np.array([G,K,M,G]),nk=1000)

    BZ = np.zeros([3,2])        
    BZ[0][0] = -8.51553165192668
    BZ[1][0] = 8.515531651926683
    BZ[2][0] = 17.03106487517293
    BZ[0][1] = -14.749335289001465
    BZ[1][1] = -14.749335289001465
    BZ[2][1] = 0.0
    
    points_py = grid_maker([BZ[0][0],BZ[0][1]],[BZ[2][0],BZ[2][1]],grid_num)
    kx = points_py[:,0]
    ky = points_py[:,1]

    os.mkdir(f'{N}_faults_conv_{grid_num}_{in_list}')
    d_normed, pdos_normed = Bands_DOS(D,t0,t1,t3,t4,N,in_list,[kx_band,ky_band],[kx,ky],np.linspace(-E,E,E_num),alpha)
    np.savetxt(f'{N}_faults_conv_{grid_num}_{in_list}/d_normed.dat',d_normed,delimiter=';')
    np.savetxt(f'{N}_faults_conv_{grid_num}_{in_list}/pdos_normed.dat',pdos_normed,delimiter=';')    




if __name__ == '__main__':
    D = -0.077
    t0 = -2.59
    t1 = -0.336
    t3 = -0.2655
    t4 = -0.146
    E = 1
    E_num = 1001
    alpha = 0.01
    grid_num = 720
    for N in range(2,11):
        for in_list in ossz_kombok_unzipped[N]:
            main(D,t0,t1,t3,t4,N,in_list,E,E_num,grid_num,alpha)


