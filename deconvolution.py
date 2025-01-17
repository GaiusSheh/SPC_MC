import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.figsize'] = (4,3)
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['lines.linewidth'] = 1.0

def test_func(ADU):
    if ADU < 100:
        prob = 0
    else:
        prob = (ADU - 100) ** 0.5 * np.exp(-(ADU - 0) / 30)
    prob += 1 * np.exp(-((ADU - 200)) ** 2 / 25)
    return prob


def convolution(a,b):
    ans = 0
    if a.shape[0] == b.shape[0]:
        length = a.shape[0]
        ans = np.convolve(a,b,mode='full')[0:length]
    else:
        print('array length error')
    return ans

def Auto_Convolution(f,n):
    ans = []
    if n == 1:
        ans = f
    if n > 1:
        g = Auto_Convolution(f,n-1)
        #print('g:',g)
        ans = convolution(f,g)
    return np.array(ans)

def A(m):
    return ((-1) ** (m-1)) * (1/ m)

def reconstruct(I,Poisson_Lambda,order):
    ans = I *0
    for n in range(1,order):
        ans +=  (A(n)/Poisson_Lambda) * np.exp(n * Poisson_Lambda) * Auto_Convolution(I,n)
    return ans






###############################################################################################################################################

###############################################################################################################################################

###############################################################################################################################################



run_num = [14]

out_path = Path("MC_charge_spread/SPC_outputs")
for run in run_num:
    order = 5

    Observed = np.load(out_path / f"{str(run).zfill(4)}_hist.npy")[0]
    ADU_grid = np.load(out_path / f"{str(run).zfill(4)}_hist.npy")[1]
    # print(Observed)
    # print(ADU_grid)
    original = np.array([test_func(ADU) for ADU in ADU_grid])   
    # print(original)
    

    λ_set = [0.05, 0.06, 0.07]

    ob_not_plotted = True
    for λ in λ_set:
        N = np.sum(Observed)
        print('N:',N)
        Normalisation_factor = N /(λ*np.exp(-λ))
        Observed /= Normalisation_factor
        print(λ,'Observed Integral:',np.sum(Observed))
        # print('Observed:',Observed)
        reconstructed = reconstruct(Observed,λ,order)
        print(λ,'Reconstrcuted Integral:',np.sum(reconstructed))
        # print('Reconstructed:',reconstructed)

        if ob_not_plotted:
            plt.plot(ADU_grid,Observed/np.sum(Observed),label='Observed')
            plt.plot(ADU_grid,original/np.sum(original),label='True')
            ob_not_plotted = False
        plt.plot(ADU_grid,reconstructed/np.sum(reconstructed),label=f'$\\beta_{{\\mathrm{{Pois}}}} = {λ}$')
        #plt.yscale('log')

    plt.legend()
    plt.xlabel('ADU')
    plt.ylabel('Intensity (a.u.)')
    plt.xlim(250,450)
    plt.ylim(-1e-4,9e-4)
    plt.tight_layout()
    plt.show()
        



    

    
    



