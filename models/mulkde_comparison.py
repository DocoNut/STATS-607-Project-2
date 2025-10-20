import numpy as np
import matplotlib.pyplot as plt
from models.other_kde import plugin_kde,adaptive_kde
from models.data_preparation import DistributionSampler
from models.mulkde import kde,multi_kde_n0 as multi_kde

def comparison():
    """main function"""
    N = 100
    h = N**(-0.2)

    # F Distribution
    # dfn = 10
    # dfd = 14
    # sampler = DistributionSampler('f',[dfn,dfd])

    #beta
    # d1 = 1.3
    # d2 = 2.5
    # sampler = DistributionSampler('beta',[d1,d2])

    # bimodal
    mu1 = -2
    mu2 = 1.5
    s1 = 0.6
    s2 = 0.4
    p = 0.6
    sampler = DistributionSampler('bimodal',[mu1,mu2,s1,s2,p])

    # generate data and true pdf
    data = sampler.generate_samples(N) 
    f = sampler.get_pdf()
        
    # several estimated pdf  
    f_es = kde(h,data)
    f_ps = plugin_kde(h,data)
    f_as = adaptive_kde(h,data)
    xii = np.sqrt(np.linspace(1,4,4))
    f_xxi = multi_kde(h,data,xii,d=4)

    # generate points for plotting
    x = np.linspace(-4,4,100)
    y = f(x)
    y_es = f_es(x)
    y_ps = f_ps(x)
    y_as = f_as(x)
    y_xxi = f_xxi(x)

    # compute and compare IMSE
    dx = x[1]-x[0]
    imse_mke = np.sum((y_xxi-y)**2)*dx
    imse_adp = np.sum((y_as-y)**2)*dx
    imse_plugin = np.sum((y_ps-y)**2)*dx
    print("MKDE:",imse_mke)
    print("DDE:",imse_plugin)
    print("AKDE:",imse_adp)

    # plot these methods
    plt.figure()
    plt.plot(x,y,label='Real',color='blue')
    plt.plot(x,y_es,label='KDE',color='red')
    plt.plot(x,y_ps,label='DDE',color='purple')
    plt.plot(x,y_as,label='AKDE',color='yellow')
    plt.plot(x,y_xxi,label=f'MKDE',color='green')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    comparison()