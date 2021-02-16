<h1 align="center">
	Diabetic Retinopathy Classification Using DNN 
</h1>

### *Poster*: [Diabetic Retinopathy Classification Using DNN](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/Poster_Diabetic_Retionopathy_.pdf)
### **For Running the  code directly using [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tnioi5B035QGOyrtyuihTkdxr1zWXGsA#scrollTo=mzZlrU-lswsG)**

1. [Input Pipeline](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/input_pipeline/datasets2.py#L88)- (Training set Images- 413, Test set Images-103)
    * Image Resize (to 256*256)
    * Image crop (Box crop)
    * Image Normalization 
    * Class balancing
2. [Model Architecture](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/models/architecture.py#L22)
3. [Training Routine](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/main.py#L100)
4. Model CallBacks:
    * [Check point Callback](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/main.py#L84) - For saving model at desired interval(epoch frequency)
    * [Tensorboard Callback](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/main.py#L70) - For logging training stats,Profiling
    * [CSV Logger Callback](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/main.py#L95) - To save training logs in a csv file
5. [Training from a check point](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/main.py#L104)
    * Initial epoch here is the point at which the training was interrupted
6. [Evaluation](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/evaluation/eval.py#L35)
    * [Confusion Matrix]()
    * [Classification Report]()
7. [Data Augmentation](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/input_pipeline/datasets2.py#L45)
    * Vertical Flip
    * Horizontal Flip
    * Box Crop
    * Rotate
8. [Deep Visualization](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/visualization/deep_vis.py)
    * GradCAM
9. [Hyper Parameter Tuning](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/hyper_parameter_tuning/hparam_tuning.py)
    * [Grid Search](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/hyper_parameter_tuning/hparam_tuning.py#L22) (Epochs, Number of dense neurons, stride, Learning rate)
## Outputs from several stages of project
* **After Image processing and data augmentation:**
<p align="center">
  <img src="https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/media/9x9_aug.png" width="650" height="650"/>
</p>
<p align="center">
    <em>Processed and Augmented Images</em>
</p>

* **Model Architecture**
	1. [Model](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/media/fullmodel_tl_82_acc.png) based on ResNET50v2 (Transfer 				learning). 
	2. Model based on  Blocks of Conv+BatchNorm+Maxpool (Only 453K parameters)
	<p align="center">
	  <img src="https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/media/model_architecture.PNG" height="200"/>
	</p>
	<p align="center">
	    <em>Custom Architecture based on blocks of Conv+BatchNorm+Maxpool (ii)</em>
	</p>


	
* **Training Results**
	* Model based on  Blocks of Conv+BatchNorm+Maxpool
	<p align="center">
	  <img src="https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/media/custom_model_acc.svg" height="200"/>
	  <img src="https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/media/TL_epoch_accuracy.svg" height="200"/>
	  <img src="https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/media/legend.PNG" />
	</p>
	<p align="center">
	    <em>Accuracy plot for Custom model (on left) and ResNET50v2 based model (on right) <b> [epochs vs accuracy] </b> </em>
	</p>		  
	
* **Results and Evaluation**
	* #### Test accuracy - 77.8% (Custom Model), 81.55% (Finetuned on ResNET50v2)
	<p align="center">
	  <img src="https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/media/confusionmatrix.png" width="580" height="440"/>
	</p>
	<p align="center">
	    <em><b>Confusion Matrix</b></em>
	</p>  
	  
	  
	 <table border="1" class="Classification report" align ='center'>	
	  <thead>	  
	    <tr style="text-align: right;">
	      <th></th>
	      <th>precision</th>
	      <th>recall</th>
	      <th>f1-score</th>
	      <th>Support</th>
	    </tr>
	  </thead>
	  <tbody>
	    <tr>
	      <th>NRDR</th>
	      <td>0.73</td>
	      <td>0.82</td>
	      <td>0.77</td>
	      <td>39</td>
	    </tr>
	    <tr>
	      <th>RDR</th>
	      <td>0.88</td>
	      <td>0.81</td>
	      <td>0.85</td>
	      <td>64</td>
	    </tr>
	    <tr>
	      <th></th>
	      <td></td>
	      <td></td>
	      <td></td>
	      <td></td>
	    </tr>		  
	    <tr>
	      <th>accuracy</th>
	      <td>0.82</td>
	      <td>0.82</td>
	      <td>0.82</td>
	      <td>103</td>
	    </tr>
	    <tr>
	      <th>macro avg</th>
	      <td>0.80</td>
	      <td>0.82</td>
	      <td>0.81</td>
	      <td>103</td>
	    </tr>
	    <tr>
	      <th>weighted avg</th>
	      <td>0.82</td>
	      <td>0.82</td>
	      <td>0.82</td>
	      <td>103</td>
	    </tr>
	  </tbody>
	</table>
	<p align="center">
	    <em><b>Classification Report</b></em>
	</p>	
	
* **Deep Visualization**

	<p align="center">
	  <img src="https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/media/vis_without_axis.png" height="200"/>
	</p>
	<p align="center">
	    <em><b>Original Image, GradCAM output, Overlay</b></em>
	</p>  
* **Hyperparameter optimization**

	<p align="center">
	  <img src="https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/media/Capture.PNG"  width="580" height="300"/>
	</p>
	<p align="center">
	    <em><b>Original Image, GradCAM output, Overlay</b></em>
	</p> 
	
## Instructions to run the script:

Before running the script Install the [requirments](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/requirements.txt) from ```requirements.txt``` using ```pip install -r requirements.txt```

* **Make the following changes in main.py based on the Mode(training mode, hyper parameter tuning mode, finetuning mode, evaluation mode) you want you the script in**

	1. To Train the model, change the train FLAG in ```main.py``` to ```True```  
		```flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')```  
		* To log the data , specify path to tensorboard callback, model chekpoint call back, CSVlogger call back in ```constants.py```
			```dir_all_logs = 'log_dir'```  
			```dir_fit = os.path.join(dir_all_logs, 'fit') ```    
			```dir_cpts = os.path.join(dir_all_logs, 'cpts') ```    
			```dir_csv = os.path.join(dir_all_logs, 'csv_log') ```    
			   		
	2. For performing hyperparamter Tuning  

		```flags.DEFINE_boolean('hparam_tune', True, 'Specify if its hyper param tuning.')```  

	3. For Training the model based on ResNET50v2

		```flags.DEFINE_boolean('Transfer_learning', True, 'to use transfer learning based model, train flag must be set to true to fine tune pretrained model')```	

	4. For Evaluating the pretrained model 
		* Change the path of the pretrained model [here](https://github.com/sgaruda-sudo/Diabetic_Retinopathy/blob/master/main.py#L152) in ```main.py``` to desired path.

		```flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')```


