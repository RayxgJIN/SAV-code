from distutils.dep_util import newer_pairwise
from logging import WARN
from re import S
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

N_V = 4
QUAD_ORDER = 2
QUAD_CORD, QUAD_WGHT = np.polynomial.legendre.leggauss(QUAD_ORDER)
N = 30
steps = 10
L=1
T=1


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
        #return np.cos(np.pi*x)*np.cos(np.pi*y)*np.exp(-t)
        return np.cos(t)*np.cos(np.pi*x)*np.cos(np.pi*y)

def t_ref_phi_func(x,y,t):
        #return np.cos(np.pi*x)*np.cos(np.pi*y)*np.exp(-t)
        return -np.sin(t)*np.cos(np.pi*x)*np.cos(np.pi*y)
    
def ref_theta_func(x,y,thetac,t):
        #return np.cos(np.pi*x)*np.cos(np.pi*y)*np.exp(-t)+thetac
        return np.sin(t)*np.sin(np.pi*x)*np.sin(np.pi*y)+thetac
def t_ref_theta_func(x,y,thetac,t):
        #return np.cos(np.pi*x)*np.cos(np.pi*y)*np.exp(-t)+thetac
        return np.cos(t)*np.sin(np.pi*x)*np.sin(np.pi*y)

elem_Lap_stiff_mat = np.zeros((N_V, N_V))
for loc_ind_i in range(N_V):
    for loc_ind_j in range(N_V):
        elem_Lap_stiff_mat[loc_ind_i, loc_ind_j] = get_loc_stiff(1.0, loc_ind_i, loc_ind_j)


class Solver:
    lambda_ = 1
    alpha = 1
    gamma = 1
    delta = 1.2
    h = 0.0
    tau = 0.0
    epsilion = 1
    thetac=0
    dof_num = 0

    def __init__(self, N: int, L: int):
        self.N = N
        self.h = L /N
        # phi 0::2, theta 1::2
        self.dof_num = 2 * (self.N + 1)**2
        
    def W_func(self, x):
        return 0.25 * (x**2 - 1)**2

    def W_prime_func(self, x):
        return x**3 - x

    def new_W_prime_func(self, x):
        return (x**3 - x)*x

    def p_prime_func(self, x):
        return -0.5

    def get_L2_energy_norm(self, u):
        val0 = self.h * np.linalg.norm(u)
        return val0

    def get_next_step(self, v, q, Time, t):

        self.tau= Time
        
        self.t=t

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
                    location_i=elem_ind_x+loc_node_ind_row_x
                    location_j=elem_ind_y+loc_node_ind_row_y
          

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
                    C[node_ind_row * 2] =   0.25*self.h**2 * self.alpha /self.tau * v[node_ind_row * 2]
                    C[node_ind_row * 2 ]+=  0.25*self.h**2 * self.gamma *self.thetac * self.p_prime_func(v[node_ind_row * 2])
                    w[node_ind_row * 2 ]=  0.25*self.h**2 * self.W_prime_func(v[node_ind_row * 2])/(self.epsilion * Q_h)
                #C[node_ind_row * 2 ]+= 0.25* self.h**2 * self.gamma * v[node_ind_col * 2+1] * self.p_prime_func(v[node_ind_col * 2])
                #C[node_ind_row * 2 ] -= self.lambda_ * q * 0.25* self.h**2 * self.W_prime_func(v[node_ind_col * 2])/(self.epsilion * Q_h) 
                    C[node_ind_row * 2] +=  0.25* self.h**2 * (-self.alpha* t_ref_phi_func(location_i*self.h,location_j*self.h,self.t)+self.lambda_*self.epsilion*(-2*np.pi**2)*ref_phi_func(location_i*self.h,location_j*self.h,self.t)-self.lambda_*(1/self.epsilion)* self.W_prime_func(ref_phi_func(location_i*self.h,location_j*self.h,self.t))+0.5*self.gamma*(ref_theta_func(location_i*self.h,location_j*self.h,self.thetac,self.t)-self.thetac))
                #C[node_ind_row * 2 ] -= 0.25* self.h**2 * (self.alpha * ref_phi_func(location_i*self.h,location_j*self.h,self.tau)+self.lambda_*self.epsilion*(-2*np.pi**2)*ref_phi_func(location_i*self.h,location_j*self.h,self.tau)-self.lambda_*(1/self.epsilion)* self.W_func(ref_phi_func(location_i*self.h,location_j*self.h,self.tau))+0.5*self.gamma*(ref_theta_func(location_i*self.h,location_j*self.h,self.thetac,self.tau)-self.thetac))
                    

                #R_{theta}
                    C[node_ind_row * 2 + 1] =  0.25* self.h**2 * self.delta/self.tau * v[node_ind_row* 2 + 1] 
                    C[node_ind_row * 2 + 1] -=  0.25* self.h**2 * self.gamma /self.tau * v[node_ind_row * 2] 
                #C[node_ind_row * 2 + 1] -=  0.25* self.h**2* (self.delta * ref_theta_func(location_i*self.h,location_j*self.h,self.thetac, self.tau)+ self.gamma * 0.5 * ref_phi_func(location_i*self.h,location_j*self.h, self.tau)-2*np.pi**2*ref_theta_func(location_i*self.h,location_j*self.h,self.thetac, self.tau))
                    C[node_ind_row * 2 + 1] +=  0.25* self.h**2 * (-self.delta * t_ref_theta_func(location_i*self.h,location_j*self.h, self.thetac, self.t)- self.gamma * 0.5 * t_ref_phi_func(location_i*self.h,location_j*self.h, self.t)-2*np.pi**2*ref_theta_func(location_i*self.h,location_j*self.h,self.thetac, self.t))
                #C[node_ind_row*2] -=0.25* self.h**2 * (-self.alpha* t_ref_phi_func(location_i*self.h,location_j*self.h,self.tau)+self.lambda_*self.epsilion*(-2*np.pi**2)*ref_phi_func(location_i*self.h,location_j*self.h,self.tau)-self.lambda_*(1/self.epsilion)* self.W_prime_func(ref_phi_func(location_i*self.h,location_j*self.h,self.tau))+0.5*self.gamma*(ref_theta_func(location_i*self.h,location_j*self.h,self.thetac,self.tau)-self.thetac))
                    

        A_mat_coo = coo_matrix((VV[:flag], (II[:flag], JJ[:flag])), shape=(self.dof_num, self.dof_num))
        A_mat = A_mat_coo.tocsc()
        C= C - self.lambda_* newq * w
        
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
        nnq = np.sqrt(1.0 / epsilion * np.sum(h**2 * W_phi) + 1.0)      
  

    #Compare phi, theta and real_phi, real_theta
        realsol=np.zeros((self.dof_num, 1))
        for elem_ind in range(self.N**2):
         elem_ind_y, elem_ind_x = divmod(elem_ind, self.N)
         for loc_node_ind_row in range(N_V):
            loc_node_ind_row_y, loc_node_ind_row_x = divmod(loc_node_ind_row, 2)
            node_ind_row = (elem_ind_y + loc_node_ind_row_y) * (self.N + 1) + elem_ind_x + loc_node_ind_row_x
            location_i=elem_ind_x+loc_node_ind_row_x
            location_j=elem_ind_y+loc_node_ind_row_y
            realsol[node_ind_row * 2]= ref_phi_func(location_i*self.h,location_j*self.h, self.tau)
            realsol[node_ind_row * 2+1]=ref_theta_func(location_i*self.h,location_j*self.h, self.thetac, self.tau)

        # def get_errors():
        phierror= self.get_L2_energy_norm(v[::2]-realsol[::2])
        thetaerror= self.get_L2_energy_norm(v[1::2]-realsol[1::2])
        
        return phierror, thetaerror, v, nnq


#solver = Solver(N, steps, 1.0, 1.0)
#for i in range(1, solver.steps+1):
# time = i * solver.tau
# err_phi_list[k], err_theta_list[k]= solver.get_next_step(inisol, q) 


#Plot the error
K=4
t_new=[0.0] * K
err_phi_list = [0.0] * K
err_theta_list = [0.0] * K
new_err_phi_list=[0.0] *K
for j in range (K):
    SEC_NUM=steps*(2**(j+1))
    #Get initial solution of phi, theta and q
    epsilion = 1      
    dof_num= 2 * (N + 1)**2
    h=1/N
    thetac=0
    inisol=np.zeros((dof_num, 1))
    for elem_ind in range(N**2):
        elem_ind_y, elem_ind_x = divmod(elem_ind, N)
        for loc_node_ind_row in range(N_V):
          loc_node_ind_row_y, loc_node_ind_row_x = divmod(loc_node_ind_row, 2)
          node_ind_row = (elem_ind_y + loc_node_ind_row_y) * (N + 1) + elem_ind_x + loc_node_ind_row_x
          location_i=elem_ind_x+loc_node_ind_row_x
          location_j=elem_ind_y+loc_node_ind_row_y
          inisol[node_ind_row * 2]= ref_phi_func(location_i*h, location_j*h, 0)
          inisol[node_ind_row * 2+1]= ref_theta_func(location_i*h, location_j*h, thetac, 0)

    phi_part= np.zeros(((N + 1)**2, 1))
    phi_part = inisol[::2]
    phi_part = phi_part.reshape((N + 1, N + 1))
    W_phi = 0.25*(1-phi_part**2)**2
    W_phi[0, :] *= 0.5
    W_phi[-1, :] *= 0.5
    W_phi[:, 0] *= 0.5
    W_phi[:, -1] *= 0.5
    q = np.sqrt(1.0 / epsilion * np.sum(h**2 * W_phi) + 1.0)

    t=[0.0] * SEC_NUM
    err_phi = [0.0] * SEC_NUM
    err_theta = [0.0] * SEC_NUM
    

#ela_list = [0.0] * SEC_NUM 
    for k in range(SEC_NUM):
       t[k]=(k+1)* (1/SEC_NUM)
       solver = Solver(N, L)
       err_phi[k], err_theta[k], inisol, q = solver.get_next_step(inisol, q, 1/SEC_NUM, t[k])
    t_new[j], err_phi_list[j], err_theta_list[j] =1/SEC_NUM, np.max(err_phi), np.max(err_theta)


print(t_new, err_phi_list, err_theta_list)
#Plot the error norm
plt.plot(t_new, err_phi_list, 'b-o', label='$\\varphi$')
plt.xlabel('Time step size')
plt.ylabel('Error norm')
plt.legend()
plt.show()
plt.plot(t_new, err_theta_list, 'g-^', label='$\\theta$')
plt.xlabel('Time step size')
plt.ylabel('Error norm')
plt.legend()
plt.show()



