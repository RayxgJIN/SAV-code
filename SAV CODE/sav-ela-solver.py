from distutils.dep_util import newer_pairwise
from logging import WARN
from re import S
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from itertools import product


N_V = 4
QUAD_ORDER = 2
QUAD_CORD, QUAD_WGHT = np.polynomial.legendre.leggauss(QUAD_ORDER)
N = 50
steps = 10000
L=1
T=1

DIM = 2
QUAD_ORDER = 2
QUAD_PNTS = QUAD_ORDER**DIM
QUAD_CORD, QUAD_WGHT = np.polynomial.legendre.leggauss(QUAD_ORDER)


def get_locbase_val(loc_ind: int, x: float, y: float):
    val = -1.0
    if loc_ind == 0:
        val = 0.25 * (1.0 - x) * (1.0 - y)
    elif loc_ind == 1:
        val = 0.25 * (1.0 + x) * (1.0 - y)
    elif loc_ind == 2:
        val = 0.25 * (1.0 - x) * (1.0 + y)
    elif loc_ind == 3:
        val = 0.25 * (1.0 + x) * (1.0 + y)
    else:
        raise ValueError("Invalid option")
    return val


def get_locbase_grad_val(loc_ind: int, x: float, y: float):
    grad_val_x, grad_val_y = -1.0, -1.0
    if loc_ind == 0:
        grad_val_x = -0.25 * (1.0 - y)
        grad_val_y = -0.25 * (1.0 - x)
    elif loc_ind == 1:
        grad_val_x = 0.25 * (1.0 - y)
        grad_val_y = -0.25 * (1.0 + x)
    elif loc_ind == 2:
        grad_val_x = -0.25 * (1.0 + y)
        grad_val_y = 0.25 * (1.0 - x)
    elif loc_ind == 3:
        grad_val_x = 0.25 * (1.0 + y)
        grad_val_y = 0.25 * (1.0 + x)
    else:
        raise ValueError("Invalid option")
    return grad_val_x, grad_val_y


def get_loc_stiff(loc_coeff: float, loc_ind_i: int, loc_ind_j: int):
    val = 0.0
    for quad_ind_x in range(QUAD_ORDER):
        for quad_ind_y in range(QUAD_ORDER):
            quad_cord_x, quad_cord_y = QUAD_CORD[quad_ind_x], QUAD_CORD[quad_ind_y]
            quad_wght_x, quad_wght_y = QUAD_WGHT[quad_ind_x], QUAD_WGHT[quad_ind_y]
            grad_val_ix, grad_val_iy = get_locbase_grad_val(loc_ind_i, quad_cord_x, quad_cord_y)
            grad_val_jx, grad_val_jy = get_locbase_grad_val(loc_ind_j, quad_cord_x, quad_cord_y)
            val += loc_coeff * (grad_val_ix * grad_val_jx + grad_val_iy * grad_val_jy) * quad_wght_x * quad_wght_y
    return val


def ref_phi_func(x,y,t):
        return np.cos(t)*np.cos(np.pi*x)*np.cos(np.pi*y)

def t_ref_phi_func(x,y,t):
        return -np.sin(t)*np.cos(np.pi*x)*np.cos(np.pi*y)
    
def ref_theta_func(x,y,thetac,t):
        return np.sin(t)*np.sin(np.pi*x)*np.sin(np.pi*y)+thetac

def t_ref_theta_func(x,y,thetac,t):
        return np.cos(t)*np.sin(np.pi*x)*np.sin(np.pi*y)


elem_Lap_stiff_mat = np.zeros((N_V, N_V))
for loc_ind_i in range(N_V):
    for loc_ind_j in range(N_V):
        elem_Lap_stiff_mat[loc_ind_i, loc_ind_j] = get_loc_stiff(1.0, loc_ind_i, loc_ind_j)

elem_ela_rhs = np.zeros((2 * N_V,))
for loc_ind in range(N_V):
        for quad_ind_x, quad_ind_y in product(range(QUAD_ORDER), range(QUAD_ORDER)):
             quad_cord_x, quad_wght_x = QUAD_CORD[quad_ind_x], QUAD_WGHT[quad_ind_x]
             quad_cord_y, quad_wght_y = QUAD_CORD[quad_ind_y], QUAD_WGHT[quad_ind_y]
             grad_val_x, grad_val_y = get_locbase_grad_val(loc_ind, quad_cord_x, quad_cord_y)
             elem_ela_rhs[2 * loc_ind] += grad_val_x * quad_wght_x * quad_wght_y
             elem_ela_rhs[2 * loc_ind + 1] += grad_val_y * quad_wght_x * quad_wght_y

# Add by YCQ 2023-10-6
base_grad_val_at_quad_pnt = np.zeros((N_V, DIM, QUAD_PNTS))
base_val_at_quad_pnt = np.zeros((N_V, QUAD_PNTS))
quad_wghts = np.zeros((QUAD_PNTS,))
for loc_nd_ind in range(N_V):
    for quad_pnt_ind_x in range(QUAD_ORDER):
        for quad_pnt_ind_y in range(QUAD_ORDER):
            quad_pnt_ind = quad_pnt_ind_y * QUAD_ORDER + quad_pnt_ind_x
            x, y = QUAD_CORD[quad_pnt_ind_x], QUAD_CORD[quad_pnt_ind_y]
            base_grad_val_at_quad_pnt[loc_nd_ind, :, quad_pnt_ind] = get_locbase_grad_val(loc_nd_ind, x, y)
            base_val_at_quad_pnt[loc_nd_ind, quad_pnt_ind] = get_locbase_val(loc_nd_ind, x, y)
            quad_wghts[quad_pnt_ind] = QUAD_WGHT[quad_pnt_ind_x] * QUAD_WGHT[quad_pnt_ind_y]


def get_loc_stiff_mat(E, nu):
    loc_Amat = np.zeros((2 * N_V, 2 * N_V))
    lamb_3d = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    for m in range(N_V):
        for n in range(N_V):
            fdm_m, fdm_n = m * DIM, n * DIM  # m'=0, n'=0
            loc_Amat[fdm_m, fdm_n] += (lamb_3d + 2 * mu) * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
            # loc_Amat[fdm_m, fdm_n] += 0.5 * mu * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
            loc_Amat[fdm_m, fdm_n] += mu * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)

            fdm_m, fdm_n = m * DIM, n * DIM + 1  # m'=0, n'=1
            loc_Amat[fdm_m, fdm_n] += lamb_3d * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
            # loc_Amat[fdm_m, fdm_n] += 0.5 * mu * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
            loc_Amat[fdm_m, fdm_n] += mu * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)

            fdm_m, fdm_n = m * DIM + 1, n * DIM  # m'=1, n'=0
            loc_Amat[fdm_m, fdm_n] += lamb_3d * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
            # loc_Amat[fdm_m, fdm_n] += 0.5 * mu * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
            loc_Amat[fdm_m, fdm_n] += mu * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)

            fdm_m, fdm_n = m * DIM + 1, n * DIM + 1  # m'=1, n'=1
            loc_Amat[fdm_m, fdm_n] += (lamb_3d + 2 * mu) * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
            # loc_Amat[fdm_m, fdm_n] += 0.5 * mu * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
            loc_Amat[fdm_m, fdm_n] += mu * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
    return loc_Amat

class Solver:
    lambda_ = 1
    alpha = 1
    gamma = 1
    delta = 1.2
    epsilion = 0.001
    thetac=0


    # Add by YCQ 2023-10-6
    beta=1.0
    kappa = 1.0e-3
    E = 2.0
    poisson_ratio = 0.4
    p_gel = 0.2
   

    def __init__(self, N: int, L: int):
        self.N = N
        self.L=L
        self.h = self.L / N

        # phi 0::2, theta 1::2
        self.dof_num = 2 * (self.N + 1)**2
        self.dof_ela_num = 2 * (self.N - 1)**2
        self.lamb_3d = self.E * self.poisson_ratio / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
        self.mu = self.E / (2.0 * (1.0 + self.poisson_ratio))


    def W_func(self, x):
        return 0.25 * (x**2 - 1)**2

    def W_prime_func(self, x):
        return x**3 - x

    def p_prime_func(self, x):
        if -1 <= x <= 1:
            return -1.0
        else:
            return 0.0
    
    def new_W_prime_func(self, x):
        return (x**3 - x)*x

    def k(self, phi):
        if phi <= self.p_gel:
            return 0.0
        else:
            return (phi - self.p_gel) / (1.0 - self.p_gel)

    def m(self, x):
       return x

    def get_L2_energy_norm(self, u):
        val0 = self.h * np.linalg.norm(u)
        return val0

    
    def get_next_step(self, v, q, Time):

        self.tau= Time
        
        # Get Q_h
        phi=v[::2]
        phi_part= phi.reshape((self.N + 1), (self.N + 1))
        W_phi = self.W_func(phi_part)
        W_phi[0, :] *= 0.5
        W_phi[-1, :] *= 0.5
        W_phi[:, 0] *= 0.5
        W_phi[:, -1] *= 0.5
        Q_h = np.sqrt(1.0 / self.epsilion * np.sum(self.h**2 * W_phi) + 1.0)
        
        new_W_pre_phi = self.new_W_prime_func(phi_part)
        new_W_pre_phi[0, :] *= 0.5
        new_W_pre_phi[-1, :] *= 0.5
        new_W_pre_phi[:, 0] *= 0.5
        new_W_pre_phi[:, -1] *= 0.5
        newq = q- np.sum(self.h**2 * new_W_pre_phi) /(2 * self.epsilion * Q_h)



      # Generate Mat A, Vec c, Vec w
        max_entries = self.N**2 * 64
        II = -np.ones(max_entries, dtype=np.int32)
        JJ = -np.ones(max_entries, dtype=np.int32)
        VV = np.zeros(max_entries)
        
        C = np.zeros((self.dof_num , 1))
        w = np.zeros((self.dof_num, 1))
        P = np.zeros((self.dof_num, 1))
        flag = 0

        for elem_ind in range(self.N**2):
            elem_ind_y, elem_ind_x = divmod(elem_ind, self.N)
            for loc_node_ind_row in range(N_V):
                    loc_node_ind_row_y, loc_node_ind_row_x = divmod(loc_node_ind_row, 2)
                    node_ind_row = (elem_ind_y + loc_node_ind_row_y) * (self.N + 1) + elem_ind_x + loc_node_ind_row_x

          

                    for loc_node_ind_col in range(N_V):
                        loc_node_ind_col_y, loc_node_ind_col_x = divmod(loc_node_ind_col, 2)
                        node_ind_col = (elem_ind_y + loc_node_ind_col_y) * (self.N + 1) + elem_ind_x + loc_node_ind_col_x
                    

                      # A_{phi, phi}
                        II[flag] = node_ind_row * 2
                        JJ[flag] = node_ind_col * 2
                        VV[flag] = self.lambda_ * self.epsilion * elem_Lap_stiff_mat[loc_node_ind_row, loc_node_ind_col]
                        if loc_node_ind_row == loc_node_ind_col:
                           VV[flag] += 0.25 * self.h**2 * self.alpha / self.tau
                        flag += 1

                      # A_{theta, theta}
                        II[flag] = node_ind_row * 2 + 1
                        JJ[flag] = node_ind_col * 2 + 1
                        VV[flag] = elem_Lap_stiff_mat[loc_node_ind_row, loc_node_ind_col]
                        if loc_node_ind_row == loc_node_ind_col:
                           VV[flag] += 0.25 * self.h**2 * self.delta / self.tau
                        flag += 1

                       # A_{theta, phi}
                        II[flag] = node_ind_row * 2 + 1
                        JJ[flag] = node_ind_col * 2
                        if loc_node_ind_row == loc_node_ind_col:
                          VV[flag] += 0.25 * self.h**2 * self.p_prime_func(v[node_ind_col * 2]) * self.gamma
                        flag += 1

                       # A_{phi, theta}
                        II[flag] = node_ind_row * 2
                        JJ[flag] = node_ind_col * 2 + 1
                        if loc_node_ind_row == loc_node_ind_col:
                          VV[flag] -= 0.25 * self.h**2 * self.p_prime_func(v[node_ind_col * 2]) * self.gamma / self.tau
                        flag += 1

                # Get loc rhs
                #Combined R_{phi}
                    C[node_ind_row * 2] +=   0.25*self.h**2 * self.alpha /self.tau * v[node_ind_row * 2]
                    C[node_ind_row * 2 ]+=  0.25*self.h**2 * self.gamma *self.thetac * self.p_prime_func(v[node_ind_row * 2])
                    w[node_ind_row * 2 ]+=  0.25*self.h**2 * self.W_prime_func(v[node_ind_row * 2])/(self.epsilion * Q_h)

                #R_{theta}
                    C[node_ind_row * 2 + 1] +=  0.25* self.h**2 * self.delta/self.tau * v[node_ind_row* 2 + 1] 
                    C[node_ind_row * 2 + 1] -=  0.25* self.h**2 * self.gamma /self.tau * v[node_ind_row * 2] 
             

        A_mat_coo = coo_matrix((VV[:flag], (II[:flag], JJ[:flag])), shape=(self.dof_num, self.dof_num))
        A_mat = A_mat_coo.tocsc()
        C= C + self.lambda_* newq * w
        
    #Step 1: Calculate inv(A)C and inv(A)(w,0)^t
        #inv(A)C
        A_mat=A_mat.todense()
        AC=np.linalg.inv(A_mat) * C
        AW=np.linalg.inv(A_mat) * w

    #Step 2: Multiply (w,0)^t on both sides to obtain w^t\phi
        A_new = 1+0.5*self.lambda_* np.inner(w, AW)
        C_new= np.inner(w, AC)
        w_phi = C_new/A_new

    #Step 3: Obtain phi , theta, q
        newrhs= C - w_phi * 0.5*self.lambda_* w
        A_mat=csc_matrix(A_mat)
        newv= spsolve(A_mat,newrhs)

        phi_part = newv[::2]
        phi_part = phi_part.reshape((self.N + 1, self.N+1))
        W_phi = 0.25*(1-phi_part**2)**2
        W_phi[0, :] *= 0.5
        W_phi[-1, :] *= 0.5
        W_phi[:, 0] *= 0.5
        W_phi[:, -1] *= 0.5
        nnq = np.sqrt(1.0 / self.epsilion * np.sum(self.h**2 * W_phi) + 1.0)      
  
        
        return newv, nnq



    def get_u(self, inisol, v):
        # Generate Mat A
        flag = 0
        max_entries = self.N**2 * 64
        II = -np.ones(max_entries, dtype=np.int32)
        JJ = -np.ones(max_entries, dtype=np.int32)
        VV = np.zeros(max_entries)
        rhs = np.zeros((self.dof_ela_num , 1))

        for elem_ind in range(self.N**2):
            elem_ind_y, elem_ind_x = divmod(elem_ind, self.N)
            loc_E = [0.0, 0.0, 0.0, 0.0]
            for loc_node_ind in range(N_V):
                loc_node_ind_y, loc_node_ind_x = divmod(loc_node_ind, 2)
                node_ind_row = (elem_ind_y + loc_node_ind_y) * (self.N + 1) + elem_ind_x + loc_node_ind_x
                loc_E[loc_node_ind] = self.kappa * self.E + self.k(v[node_ind_row*2]) * (1.0 - self.kappa) * self.E
            elem_E = sum(loc_E) * 0.25
            elem_stiff_mat = get_loc_stiff_mat(elem_E, self.poisson_ratio)

         
            for loc_node_ind in range(N_V):
                loc_node_ind_y, loc_node_ind_x = divmod(loc_node_ind, 2)
                dof_phi_ind = (elem_ind_y + loc_node_ind_y) * (self.N + 1) + elem_ind_x + loc_node_ind_x
                someconstant =  0.25 * self.k(v[dof_phi_ind*2])*(self.m(v[dof_phi_ind*2])-self.beta*(v[dof_phi_ind*2+1]-inisol[dof_phi_ind*2+1])) * (1/self.kappa) * 2*( self.lamb_3d+ self.mu)
                someconstant += 0.25 * (1-self.k(v[dof_phi_ind*2]))*(self.m(v[dof_phi_ind*2])-self.beta*(v[dof_phi_ind*2+1]-inisol[dof_phi_ind*2+1])) * 2* ( self.lamb_3d+ self.mu)

            for loc_node_ind_row in range(N_V):
                loc_node_ind_row_y, loc_node_ind_row_x = divmod(loc_node_ind_row, 2)
                if not ((elem_ind_x == 0 and loc_node_ind_row_x == 0) or (elem_ind_x == self.N - 1 and loc_node_ind_row_x == 1) or (elem_ind_y == 0 and loc_node_ind_row_y == 0) or (elem_ind_y == self.N - 1 and loc_node_ind_row_y == 1)):
                    node_ind_row = (elem_ind_y + loc_node_ind_row_y - 1) * (self.N - 1) + elem_ind_x + loc_node_ind_row_x - 1
                    for loc_node_ind_col in range(N_V):
                        loc_node_ind_col_y, loc_node_ind_col_x = divmod(loc_node_ind_col, 2)
                        if not ((elem_ind_x == 0 and loc_node_ind_col_x == 0) or (elem_ind_x == self.N - 1 and loc_node_ind_col_x == 1) or (elem_ind_y == 0 and loc_node_ind_col_y == 0) or (elem_ind_y == self.N - 1 and loc_node_ind_col_y == 1)):
                            node_ind_col = (elem_ind_y + loc_node_ind_col_y - 1) * (self.N - 1) + elem_ind_x + loc_node_ind_col_x - 1
                            II[flag] = node_ind_row * 2
                            JJ[flag] = node_ind_col * 2
                            VV[flag] = elem_stiff_mat[loc_node_ind_row*2, loc_node_ind_col*2]
                            flag += 1

                            II[flag] = node_ind_row * 2 + 1
                            JJ[flag] = node_ind_col * 2
                            VV[flag] = elem_stiff_mat[loc_node_ind_row *2+1, loc_node_ind_col*2]
                            flag += 1

                            II[flag] = node_ind_row * 2
                            JJ[flag] = node_ind_col * 2 + 1
                            VV[flag] = elem_stiff_mat[loc_node_ind_row*2, loc_node_ind_col*2+1]
                            flag += 1

                            II[flag] = node_ind_row * 2 + 1
                            JJ[flag] = node_ind_col * 2 + 1
                            VV[flag] = elem_stiff_mat[loc_node_ind_row*2+1, loc_node_ind_col*2+1]
                            flag += 1

        # Generate loc RHS vector
                    

                    rhs[node_ind_row*2]+= someconstant * elem_ela_rhs[loc_node_ind_col * 2] * 0.5 * self.h**2
                    rhs[node_ind_row*2+1]+= someconstant * elem_ela_rhs[loc_node_ind_col * 2+1] * 0.5 * self.h**2
                 
        # Generate CSR mat
        U_mat_coo = coo_matrix((VV[:flag], (II[:flag], JJ[:flag])), shape=(self.dof_ela_num, self.dof_ela_num))
        U_mat = U_mat_coo.tocsc()
        U_mat=csc_matrix(U_mat)
        u= spsolve(U_mat,rhs)
          
        return u[::2],  u[1::2]

#Plot the snapshot image at t=0.2s
solver = Solver(N, L)
K=1
for j in range (K):
    SEC_NUM=steps*(2**(j+1))
    #Get initial solution of phi, theta and q
    epsilion = 0.01      
    dof_num= 2 * (N + 1)**2
    h=1/N
    thetac=0
    inisol=np.zeros((dof_num, 1))
    for elem_ind in range(N**2):
        elem_ind_y, elem_ind_x = divmod(elem_ind, N)
        for loc_node_ind_row in range(N_V):
          loc_node_ind_row_y, loc_node_ind_row_x = divmod(loc_node_ind_row, 2)
          node_ind_row = (elem_ind_y + loc_node_ind_row_y) * (N + 1) + elem_ind_x + loc_node_ind_row_x
          location_i = elem_ind_x + loc_node_ind_row_x
          location_j=elem_ind_y+loc_node_ind_row_y
          inisol[node_ind_row * 2] = ref_phi_func(location_i*h, location_j*h, 0)
          inisol[node_ind_row * 2+1] = ref_theta_func(location_i*h, location_j*h, thetac, 0)

    phi_part= np.zeros(((N + 1)**2, 1))
    phi_part = inisol[::2]
    phi_part = phi_part.reshape((N + 1, N + 1))
    W_phi = 0.25*(1-phi_part**2)**2
    W_phi[0, :] *= 0.5
    W_phi[-1, :] *= 0.5
    W_phi[:, 0] *= 0.5
    W_phi[:, -1] *= 0.5
    q = np.sqrt(1.0 / epsilion * np.sum(h**2 * W_phi) + 1.0)

    t = K * (1/SEC_NUM)
    p, q = solver.get_next_step(inisol, q, 1/SEC_NUM)
    u_xlist, u_ylist= solver.get_u(p, inisol)
       
#Plot the the ela

print(u_xlist,u_ylist)
plt.imshow(u_xlist.reshape((N - 1, N - 1)))
plt.colorbar()
plt.show()
plt.imshow(u_ylist.reshape((N - 1, N - 1)))
plt.colorbar()
plt.show()
#plt.xlabel('$u_x$')
#plt.ylabel('$u_y$')
#plt.show()



