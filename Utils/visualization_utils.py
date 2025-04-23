import numpy as np
import sys
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import h5py
import os
import shutil
from PIL import Image

def table_metrics(y_pred,y_test,std_pred,outfile,doppel=False):
	"""
	Args:
		y_pred (np.array): A (batch_size,num_params) array containing the
			mean prediction for each Gaussian
		y_test (np.array): A (batch_size,num_params) array containing the
			true value of the parameter on the test set.
		std_pred (np.array): A (batch_size,num_params) array containing the
			predicted standard deviation for each parameter.
		outfile (string): File to write latex source to
	"""
	error = y_pred - y_test
	#Add mean error/MAE/P
	mean_error = np.mean(error,axis=0)
	median_error = np.median(error,axis=0)
    # mean bias in number of std deviations
	mean_bias_std_devs = np.mean(error/std_pred,axis=0)
	median_bias_std_devs = np.median(error/std_pred,axis=0)
	#plt.text(0.73,0.05,
	#	'Mean Error: %.3f'%(mean_error),{'fontsize':fontsize},transform=ax.transAxes)
	MAE = np.median(np.abs(error),axis=0)
	mean_AE = np.mean(np.abs(error),axis=0)
	avg_percent_error = np.mean(np.abs(error)/y_test,axis=0)*100
	#plt.text(0.73,0.11,
	#	'MAE: %.3f'%(MAE),{'fontsize':fontsize},transform=ax.transAxes)
	P = np.median(std_pred,axis=0)
	#plt.text(0.73,0.16,'P: %.3f'%(P),{'fontsize':fontsize},transform=ax.transAxes)
	
    # how to do % precision?
	avg_percent_precision = np.mean(std_pred/np.abs(y_pred),axis=0)*100
	median_percent_precision = np.median(std_pred/np.abs(y_pred),axis=0)*100

	cov_masks = [np.abs(error)<=std_pred,np.abs(error)<2*std_pred,
		np.abs(error)<3*std_pred]

	#metrics = [mean_bias_std_devs,median_bias_std_devs,MAE,P,
    #    avg_percent_precision,median_percent_precision]
	metrics = [median_bias_std_devs,mean_error,median_error,MAE,P,avg_percent_error,avg_percent_precision]
		
	if outfile is not None:
		f = open(outfile,'w')
	else:
		f = sys.stdout

	f.write('\hline')
	f.write('\n')
	for i,lab in enumerate(['Median Bias (in $\sigma$)','Mean Error','Median Error',
        'MAE','Median($\sigma$)','Avg. Error','Avg Prec.']):
		f.write(lab)
		f.write(' ')

		for m in metrics[i]:
			f.write('& ')
			f.write(str(np.around(m,2)))
			f.write(' ')


		f.write(r'\\')
		f.write('\n')
		f.write('\hline')
		f.write('\n')

	# write % contained in 1,2,3 sigma
	for j,cov_mask in enumerate(cov_masks):
		f.write(r'\% contained '+str(j+1)+'$\sigma$')
		f.write(' ')

		for k in range(0,len(metrics[0])):
			f.write('& ')
			if doppel:
				f.write(str(np.sum(cov_mask[:,k])) + '/' + str(len(error)))
			else:
				f.write(str(np.around(np.sum(cov_mask[:,k])/len(error),2)))
			f.write(' ')

		f.write(r'\\')
		f.write('\n')
		f.write('\hline')
		f.write('\n')
                        
	if outfile is not None:
		f.close()
		

def plot_coverage_plain(y_pred,y_test,std_pred,parameter_names,
                        fontsize=20,show_error_bars=True,n_rows=4):
    """ Generate plots for the 1D coverage of each parameter.

    Args:
        y_pred (np.array): A (batch_size,num_params) array containing the
            mean prediction for each Gaussian
        y_test (np.array): A (batch_size,num_params) array containing the
            true value of the parameter on the test set.
        std_pred (np.array): A (batch_size,num_params) array containing the
            predicted standard deviation for each parameter.
        parameter_names ([str,...]): A list of the parameter names to be
            printed in the plots.ed.
        color_map ([str,...]): A list of at least 4 colors that will be used
            for plotting the different coverage probabilities.
        block (bool): If true, block excecution after plt.show() command.
        fontsize (int): The fontsize to use for the parameter names.
        show_error_bars (bool): If true plot the error bars on the coverage
            plot.
        n_rows (int): The number of rows to include in the subplot.
    """
    num_params = len(parameter_names)
    error = y_pred - y_test
    # Define the covariance masks for our coverage plots.
    plt.rcParams['figure.dpi'] = 200
    fig = plt.figure(figsize=(20,16),num=1)
    plt.rcParams['figure.dpi'] = 80
    fig.subplots_adjust(hspace=0.3,wspace=0.3,top=0.92)
    custom_color = colors.to_hex('slateblue')
    #custom_color_hex = scale_hex_color(custom_color,0.9)
    for i in range(len(parameter_names)):
        plt.subplot(n_rows, int(np.ceil(num_params/n_rows)), i+1)
        plt.scatter(y_test[:,i],y_pred[:,i],s=60,color=custom_color)

        # Include the correlation coefficient squared value in the plot.
        straight = np.linspace(np.min(y_test[:,i]),np.max(y_test[:,i]),10)
        plt.plot(straight, straight,color='k',zorder=100)
        plt.title(parameter_names[i],fontsize=fontsize)
        plt.ylabel('Prediction',fontsize=fontsize)
        plt.xlabel('True Value',fontsize=fontsize)

        #Add mean error/MAE/P
        ax = plt.gca()
        #mean_error = np.mean(error[:,i])
        #plt.text(0.6,0.04,
        #	'Mean Error: %.3f'%(mean_error),{'fontsize':fontsize},transform=ax.transAxes)
        MAE = np.median(np.abs(error[:,i]))
        plt.text(0.73,0.05,
            'MAE: %.3f'%(MAE),{'fontsize':15},transform=ax.transAxes)
        P = np.median(std_pred[:,i])
        plt.text(0.73,0.13,'P: %.3f'%(P),{'fontsize':15},transform=ax.transAxes)
		

def matrix_plot_from_h5(file_name,dim,save_name):
    """
    Args: 
        file_name (string): path to .h5 file storing images
        dim (int,int): Tuple of (number rows, number columns), defines shape of 
            matrix plotted
        save_name (string): Filename to save final image to
    """

    # load in .h5 file
    f = h5py.File(file_name, "r")
    f_data = f['data'][()]
    print('size of dataset: ',f_data.shape)

    # TODO: check to see if this folder already exists
    os.mkdir('intermediate_temp')

    # prevent matplotlib from showing intermediate plots
    backend_ =  mpl.get_backend() 
    mpl.use("Agg")  # Prevent showing stuff

    file_counter = 0
    completed_rows = []
    offset = 40
    for i in range(0,dim[0]):
        row = []
        for j in range(0,dim[1]):
            cropped_data = f_data[file_counter]
            # normalize data using log and min cutoff 
            norm = simple_norm(cropped_data,stretch='log',min_cut=1e-6)
        
            # create individual image using plt library
            plt.figure(dpi=300)
            plt.matshow(cropped_data,cmap='viridis',norm=norm)
            plt.axis('off')

            # save intermediate file, then read back in as array, and save to row
            intm_name = ('intermediate_temp/'+ str(file_counter)
                +'.png')
            plt.savefig(intm_name,bbox_inches='tight',pad_inches=0)
            img_data = np.asarray(Image.open(intm_name))
            plt.close()
            row.append(img_data)
            # manually iterate file index
            file_counter += 1

        # stack each row horizontally in outer loop
        # edge case: one column
        if dim[1] == 1:
            build_row = row[0]
        else:
            build_row = np.hstack((row[0],row[1]))

        if dim[1] > 2:
            for c in range(2,dim[1]):
                build_row = np.hstack((build_row,row[c]))
                
        completed_rows.append(build_row)

    # reset matplotlib s.t. plots are shown
    mpl.use(backend_) # Reset backend
    
    # clean up intermediate files
    shutil.rmtree('intermediate_temp')

    # stack rows to build final image
    # edge case: one row
    if dim[0] == 1:
        final_image = completed_rows[0]
    else:
        final_image = np.vstack((completed_rows[0],completed_rows[1]))

    if dim[0] > 2:
        for r in range(2,dim[0]):
            final_image = np.vstack((final_image,completed_rows[r]))

    # plot image and save
    plt.figure(figsize=(2*dim[1],2*dim[0]),dpi=300)
    plt.imshow(final_image)
    plt.axis('off')
    plt.savefig(save_name,bbox_inches='tight')
    plt.show()