import numpy as np
import matplotlib.pyplot as plt
import modules.trainer as trainer
import exact
import time



r = [1]
k = 2
c_length = 256
n_batches = 1024
q_length = 2
epochs = 3
prune_every = 20000
mask_length = 4
gen_seq_length = mask_length*4
seq_length = gen_seq_length + mask_length
layers = 1

betastart = 0.1
betastop = 1
n_betas = 5

betas = np.linspace(betastart,betastop,n_betas)
free_energies = np.zeros(n_betas)
heat_capacities = np.zeros(n_betas)
magnetization = np.zeros(n_betas)
abs_magnetization = np.zeros(n_betas)


## Paralellism is implemented at the trainer level, not main level.
jorge = trainer.Trainer(mask_length,r,c_length,k,q_length,layers,betastart,seq_length,n_batches)

#jorge = torch.nn.DataParallel(jorge)

for i in range(n_betas):
    beta = betas[i]
    
    jorge.anneal(beta)
    
    
    anf = exact.free_energy(beta,gen_seq_length,1)
    anc = exact.heat_capacity(beta,gen_seq_length,1)
    
    print(f"Analytical free energy is {anf:0.4f}")
    print(f"Analytical heat capacity is {anc:0.4f}")
    
    tic = time.perf_counter()
    f_array, hc_array, m_array, m_abs_array = jorge.train(epochs,n_batches,beta,prune_every)
    toc = time.perf_counter()
    
    free_energies[i] = f_array[-1]
    heat_capacities[i] = hc_array[-1]
    magnetization[i] = m_array[-1]
    abs_magnetization[i] = m_abs_array[-1]
    
    print(f"Analytical free energy is {anf:0.4f}")
    print(f"Analytical heat capacity is {anc:0.4f}")
    print(f"Training took {toc - tic:0.4f} seconds")



n_a_betas = n_betas*10

a_betas = np.linspace(betastart,betastop,n_a_betas)
a_free_energies = np.zeros(n_a_betas)
a_heat_capacities = np.zeros(n_a_betas,dtype=np.double)

for i in range(n_a_betas):
    beta = a_betas[i]
    anf = exact.free_energy(beta,gen_seq_length,1)
    anc = exact.heat_capacity(beta,gen_seq_length,1)
    
    a_free_energies[i] = anf
    a_heat_capacities[i] = anc

ax_free_energy = plt.subplot(2,2,1)
ax_heat_capacity = plt.subplot(2,2,2)
ax_magnetization = plt.subplot(2,2,3)
ax_abs_magnetization = plt.subplot(2,2,4)


ax_free_energy.plot(betas,free_energies,'bo')
ax_free_energy.plot(a_betas,a_free_energies)
ax_free_energy.set_title('Free energy')
ax_free_energy.set_xlabel(' (kB T)^-1 /J')
ax_free_energy.set_ylabel(' F/J')
ax_free_energy.legend(['IsingRNNv0.9','Exact'])


ax_heat_capacity.plot(betas,heat_capacities,'bo')
ax_heat_capacity.plot(a_betas,a_heat_capacities)
ax_heat_capacity.set_title('Heat capacity')
ax_heat_capacity.set_xlabel(' (kB T)^-1 /J')
ax_heat_capacity.set_ylabel(' C /(J/kB K)')
ax_heat_capacity.legend(['IsingRNNv0.9','Exact'])


ax_magnetization.plot(betas,magnetization,'bo')
ax_magnetization.set_title('magnetization')
ax_magnetization.set_xlabel(' (kB T)^-1 /J')
ax_magnetization.set_ylabel(' <m> ')
ax_magnetization.legend(['IsingRNNv0.9'])

ax_abs_magnetization.plot(betas,abs_magnetization,'bo')
ax_abs_magnetization.set_title('magnetization')
ax_abs_magnetization.set_xlabel(' (kB T)^-1 /J')
ax_abs_magnetization.set_ylabel(' <abs(m)> ')
ax_abs_magnetization.legend(['IsingRNNv0.9'])
