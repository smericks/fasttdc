# Takes in ground truth lensing parameters from catalog, simulates an image,
# then passes that image through a paltas trained network to produce
# a predicted mass model
from paltas.Analysis import dataset_generation, loss_functions, conv_models
from paltas.Configs.config_handler import ConfigHandler
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np

class PaltasPreds:

    def __init__(self,training_config,numpix,model_weights,
        model_norms,norm_images='log_norm'):
        """
        Args:
            training_config (string): path to .py file containing paltas image 
                generation configuration
            numpix (int): sidelength of images in pixels
            model_weights (string): path to .h5 file containing paltas model 
                weights
            model_norms (string): path to .csv file containing normalizations
                used for learning_params
            norm_images (string): flag for whether to 'log_norm' or 'norm' 
                pixels before passing to the network
        """
       
        # NOTE: hardcodings here
        self.learning_params = ['main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
                        'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
                        'main_deflector_parameters_e1','main_deflector_parameters_e2',
                        'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
                        'source_parameters_center_x','source_parameters_center_y']
        self.loss_type = 'full'
        model_type = 'xresnet34'

        # setup for paltas model object
        self.img_size = (numpix,numpix,1)
        log_learning_params = []

        num_params = len(self.learning_params+log_learning_params)

        tf.random.set_seed(12)

        if self.loss_type == 'full':
            num_outputs = num_params + int(num_params*(num_params+1)/2)
            self.loss_func = loss_functions.FullCovarianceLoss(num_params)

        elif self.loss_type == 'diag':
            num_outputs = 2*num_params
            self.loss_func = loss_functions.DiagonalCovarianceLoss(num_params)

        else:
            raise ValueError('loss_type not supported in NetworkPredictions initialization')

        # create model object
        if model_type == 'xresnet101':
            self.model = conv_models.build_xresnet101(self.img_size,num_outputs)
        if model_type == 'xresnet34':
            self.model = conv_models.build_xresnet34(self.img_size,num_outputs)
        else:
            raise ValueError('model_type not supported in NetworkPredictions initialization')
        
        self.model.load_weights(model_weights)
        self.norm_path = model_norms

        # pixel norms
        self.norm_images = norm_images

        # setup paltas image generator
        self.config_handler = ConfigHandler(training_config)
    
    def create_image(self,lens_df_row):
        """
        Args:
            metadata_row(dictionary): a row from om10_metadata_venkatraman24.csv
                (a metadata.csv row from paltas)
        Returns: 
            image, metadata
        """
        # draw random sample of the nuisances
        self.config_handler.draw_new_sample()
        # TODO: update config_handlrer current sample with lens_df_row values
        # self.config_handler.sample contains a dict in form sample['main_deflector_parameters'][]

        # LENS MASS
        for key in ['theta_E','gamma1','gamma2','gamma','e1','e2',
            'center_x','center_y','z_lens']:
             self.config_handler.sample['main_deflector_parameters'][key] = (
                lens_df_row['main_deflector_parameters_'+key])
             
        # LENS LIGHT
        for key in ['R_sersic','center_x','center_y','e1','e2','mag_app',
                    'n_sersic','z_source']:
             self.config_handler.sample['lens_light_parameters'][key] = (
                lens_df_row['lens_light_parameters_'+key])
             
        # SOURCE
        for key in ['R_sersic','center_x','center_y','e1','e2','mag_app',
                    'n_sersic','z_source']:
             self.config_handler.sample['source_parameters'][key] = (
                lens_df_row['source_parameters_'+key])
             
        # POINT SOURCE
        for key in ['mag_app','x_point_source','y_point_source','z_point_source']:
             self.config_handler.sample['point_source_parameters'][key] = (
                lens_df_row['point_source_parameters_'+key])
        


        # load config file and return image that is created
        image,metadata = self.config_handler.draw_image(new_sample=False)

        return image, metadata
       

    def preds_from_params(self,lens_df):
        """Takes in catalog in paltas format
        Args:
            lens_df (pd.DataFrame)
        Returns: 
            images,metadata_list,y_pred,std_pred,cov_pred
        """
        images = []
        metadata_list = []
        num_ims = len(lens_df)
        for r in range(0,len(lens_df)):
            im, metadata = self.create_image(lens_df.loc[r])
            if im is None:
                 print('image %d failed to generate'%(r))
            # log_norm or norm images if required
            if self.norm_images == 'norm':
                im = dataset_generation.norm_image(im)
            elif self.norm_images == 'log_norm':
                im = dataset_generation.log_norm_image(im)
            images.append(im)
            metadata_list.append(metadata)
        images = np.asarray(images)
        images_tf = tf.reshape(images,[num_ims,self.img_size[0],self.img_size[1],self.img_size[2]])

        y_pred,std_pred,cov_pred = self._process_image_batch(images_tf)

        # un-normalize!!
        dataset_generation.unnormalize_outputs(self.norm_path,self.learning_params+[],  
           y_pred,standard_dev=std_pred,cov_mat=cov_pred)
        
        # need to reinforce symmetry, due to issue with numerics when un-normalizing
        # average the matrix & its transpose (basically averages the off-diagonals)
        for i in range(cov_pred.shape[0]):
            cov_pred[i] = cov_pred[i, :, :]/2 + cov_pred[i, :, :].T/2
        
        return images,metadata_list,y_pred,std_pred,cov_pred


    def _process_image_batch(self,images):
        """Generate network predictions given some batch of images

        Args:
            images

        Returns:
            y_pred,std_pred,cov_mat as numpy arrays

        """

        # use unrotated output for covariance matrix
        output = self.model.predict(images)

        if self.loss_type == 'full':
            y_pred, precision_matrix, _ = self.loss_func.convert_output(output)
        else:
            y_pred, log_var_pred = self.loss_func.convert_output(output)

        # compute std. dev.
        if self.loss_type == 'full':
            cov_mat = np.linalg.inv(precision_matrix.numpy())
            std_pred = np.zeros((cov_mat.shape[0],cov_mat.shape[1]))
            for i in range(len(std_pred)):
                std_pred[i] = np.sqrt(np.diag(cov_mat[i]))
                
        else:
            std_pred = np.exp(log_var_pred/2)
            cov_mat = np.empty((len(std_pred),len(std_pred[0]),len(std_pred[0])))
            for i in range(len(std_pred)):
                cov_mat[i] = np.diag(std_pred[i]**2)

        return y_pred.numpy(),std_pred,cov_mat
    


#########################################
######## other paltas helpers ###########
#########################################

def loss_plots(log_files,training_sizes,labels,square_error=False,
               chosen_epoch=None,xscale=True,stopping_epoch=None,
               colorname="mediumseagreen"):
	"""
	Args:
		log_files [string]: list of .csv files containing training log
		training_sizes [int]: list of training set sizes
		labels [string]: list of legend labels for each file
        square_error (bool): Whether square_error is part of the log
        chosen_epoch (int or None): if not None, plots a star at the location
			of the chosen epoch
	"""
     
	color_hex = colors.cnames[colorname] 
	color_pairs = [
		[scale_hex_color(color_hex, 0.6),scale_hex_color(color_hex, 1.6)] 
	]
		
	fig, ax = plt.subplots(1, 2,figsize=(15,7))
	for i,f in enumerate(log_files):
		df = pd.read_csv(f)
		loss = df['loss'].to_numpy()
		val_loss = df['val_loss'].to_numpy()
		if stopping_epoch:
			loss = loss[:stopping_epoch]
			val_loss = val_loss[:stopping_epoch]
		steps = (np.arange(len(loss))+1) * (training_sizes[i]/512)
		lr = 5e-3 * 0.98 ** (steps / 1e4)
		ax[0].plot(steps,loss,color=color_pairs[i][0],label=labels[i]+' Training Loss')
		ax[0].plot(steps,val_loss,color=color_pairs[i][1],label=labels[i]+' Val. Loss')
		if chosen_epoch is not None:
			ax[0].scatter(steps[chosen_epoch-1],val_loss[chosen_epoch-1],
                 marker='*',color='magenta',s=100,zorder=100)
		ax[0].set_xlabel('Steps')
		ax[0].set_ylabel('Loss')
		if xscale:
			ax[0].set_xscale('log')
		ax[0].legend()
		ax[0].grid()
		ax[1].plot(lr, loss,color=color_pairs[i][0],label=labels[i]+' Training Loss')
		ax[1].plot(lr, val_loss,color=color_pairs[i][1],label=labels[i]+' Val. Loss')
		ax[1].set_xlabel('Learning Rate')
		ax[1].set_ylabel('Loss')
		ax[1].legend()
		if xscale:
			ax[1].set_xscale('log')
		ax[1].grid()
				
		print("Final Learning Rate: ", lr[-1])

	if square_error:
		fig1,ax1 = plt.subplots(1,2,figsize=(15,7))
		for i,f in enumerate(log_files):
			df = pd.read_csv(f)
			loss = df['square_error'].to_numpy()
			val_loss = df['val_square_error'].to_numpy()
			steps = (np.arange(len(loss))+1) * (training_sizes[i]/512)
			lr = 5e-3 * 0.98 ** (steps / 1e4)
			ax1[0].plot(steps,loss,color=color_pairs[i][0],label=labels[i]+' Training SE')
			ax1[0].plot(steps,val_loss,color=color_pairs[i][1],label=labels[i]+' Val. SE')
			ax1[0].set_xlabel('Steps')
			ax1[0].set_ylabel('SE')
			ax1[0].set_xscale('log')
			ax1[0].legend()
			ax1[0].grid()
			ax1[1].plot(lr, loss,color=color_pairs[i][0],label=labels[i]+' Training SE')
			ax1[1].plot(lr, val_loss,color=color_pairs[i][1],label=labels[i]+' Val. SE')
			ax1[1].set_xlabel('Learning Rate')
			ax1[1].set_ylabel('SE')
			ax1[1].legend()
			ax1[1].set_xscale('log')
			ax1[1].grid()


	return ax


def clamp(val, minimum=0, maximum=255):
    # copied from: https://thadeusb.com/weblog/2010/10/10/python_scale_hex_color/
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return int(val)

def scale_hex_color(hexstr, scalefactor):
    """
    copied from: https://thadeusb.com/weblog/2010/10/10/python_scale_hex_color/
    Scales a hex string by ``scalefactor``. Returns scaled hex string.

    To darken the color, use a float value between 0 and 1.
    To brighten the color, use a float value greater than 1.

    >>> colorscale("#DF3C3C", .5)
    #6F1E1E
    >>> colorscale("#52D24F", 1.6)
    #83FF7E
    >>> colorscale("#4F75D2", 1)
    #4F75D2
    """

    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (r, g, b)

def early_stopping_epoch(log_file,num_before_stopping=10):
	df = pd.read_csv(log_file)
	val_loss = df['val_loss'].to_numpy()

	min_val_loss = np.inf
	chosen_epoch = np.nan
	num_waited = 0
	for i,v in enumerate(val_loss):
		if v < min_val_loss:
			min_val_loss = v
			chosen_epoch = i+1 
			num_waited = 0
		else: 
			num_waited += 1
						
			if num_waited == num_before_stopping:
				break			
						
	return chosen_epoch