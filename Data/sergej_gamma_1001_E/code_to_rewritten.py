import sys
import numpy as np
from itertools import permutations
import regex as re
import os

BZ = np.zeros([3,2])        
BZ[0][0] = -8.51553165192668
BZ[1][0] = 8.515531651926683
BZ[2][0] = 17.03106487517293
BZ[0][1] = -14.749335289001465
BZ[1][1] = -14.749335289001465
BZ[2][1] = 0.0
    
norm = 1

ossz_kombok_unzipped = []
for N in range(0,11):
    faults = np.array([f'{i}b' for i in range(1,N)])
    perms = []
    kombok = []
    for i in range(int(np.ceil(N/2))):
        for perm in sorted(set(permutations(np.concatenate((-1*np.ones(i),np.ones(N-1-i)))))):
            if list(perm)[::-1] not in perms and list(-1*np.array(perm)) not in perms:
                perms.append(list(perm))
                kombok.append((faults[np.array(perm) == -1]).tolist())
    ossz_kombok_unzipped.append(kombok)    
    
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
    points_py = grid_maker([BZ[0][0],BZ[0][1]],[BZ[2][0],BZ[2][1]],grid_num)
    kx = points_py[:,0]
    ky = points_py[:,1]
    E = np.linspace(-E,E,E_num)
    
    for N in range(2,11):
        for in_list in ossz_kombok_unzipped[N]:
            if os.path.exists(f'{N}_faults_conv_{grid_num}_{in_list}') == False:
                os.mkdir(f'{N}_faults_conv_{grid_num}_{in_list}')
            else:
                continue
            Energies, w = (np.linalg.eigh(hk(D,t0,t1,t3,t4,N,in_list,[kx,ky])))
            Energies = Energies.flatten()
            w = np.transpose(w, axes=[0,2,1])
            w = abs(w.reshape(w.shape[0]*w.shape[1], w.shape[2]))**2
            pdos = norm/alpha * w.T @ np.exp(-((Energies[:,None] - E)**2)/(2*alpha**2))
            np.savetxt(f'{N}_faults_conv_{grid_num}_{in_list}/pdos_normed.dat',pdos,delimiter=';')

