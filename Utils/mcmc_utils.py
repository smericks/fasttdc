import matplotlib.pyplot as plt
import numpy as np

def analyze_chains(emcee_chain,param_labels,true_hyperparameters,
                    outfile,show_chains=False,
                    burnin=int(1e3)):
    """
    Args:
        emcee_chain (array[n_walkers,n_samples,n_params])
    """

    if show_chains:
        for i in range(emcee_chain.shape[2]):
            plt.figure()
            #indices = np.arange(0,emcee_chain.shape[1],10)
            plt.plot(emcee_chain[:,:,i].T,'.')
            plt.title(param_labels[i])
            plt.show()

    num_params = emcee_chain.shape[2]
    chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))

    labels = ['Ground Truth', 'Inferred Value', 'Bias in $\sigma$', 'Fractional Error']

    med = np.median(chain,axis=0)
    low = np.quantile(chain,q=0.1586,axis=0)
    high = np.quantile(chain,q=0.8413,axis=0)

    error = med - true_hyperparameters
    sigma = ((high-med)+(med-low))/2
    bias = error/sigma

    metrics = [true_hyperparameters,med,bias,error]

    with open(outfile,'w') as f:

        f.write('\hline')
        f.write('\n')

        for i,lab in enumerate(labels):
            f.write(lab)
            f.write(' ')

            for k,m in enumerate(metrics[i]):
                f.write('& ')
                f.write(str(np.around(m,2)))
                f.write(' ')
                if i == 1:
                    f.write('$\pm$' + str(np.around(sigma[k],2)))

            f.write(r'\\')
            f.write('\n')
            f.write('\hline')
            f.write('\n')


    for j in range(num_params):

        med = np.median(chain[:,j])
        low = np.quantile(chain[:,j],q=0.1586)
        high = np.quantile(chain[:,j],q=0.8413)
        print(param_labels[j])
        print("\t", round(med,3), "+", round(high-med,3), "-", round(med-low,3))
        error = med - true_hyperparameters[j]
        if error > 0:
            bias = error/round(med-low,3)
        else:
            bias = error/round(high-med,3)
        print("\t", "Bias in Std. Devs: ", round(bias,3))

        frac_error = error/true_hyperparameters[j]
        print("\t","Fractional Error: ",round(frac_error,3))