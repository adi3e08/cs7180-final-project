# cs7180-final-project
This repository contains code and data for the CS7180 final project.

## Group members
- Adithya Ramesh
- Rohith Kumar Senthil Kumar
- Danny Rollo

## Directory Structure

    data/raw/ : Raw data is stored here in a separate folder for each experiment.
    
    src/: 
    	create_dataset.py: Code to create the dataset for imitation learning.
    	
    	model.py: Flow matching model definition.
    	
    	utils.py: Some utility functions.
    	
    	train.py: Code to train the flow matching model.
    	
    	eval.py: Code to evaluate the flow matching model.
    
    notebooks/ :
    	
    	run.ipynb: Notebook for colab training.
    
    models/: Trained models are stored here in a separate folder for each experiment.
    
    docs/: Documentation.
    
    results/: Tensorboard plots are stored here in a separate folder for each experiment. 

## Instructions
1. Create a local conda environment

	``` 
	conda create -n cs7180 python=3.13

	conda activate cs7180

	pip install numpy

	pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

	pip install tensorboard

	pip install matplotlib

	pip install tqdm

	pip install metaworld

	pip install scikit-learn
	```

2. Create the dataset for imitation learning (locally)
   
   ``` python -m src.create_dataset ```

3. Train the flow matching model 
	
	To train locally: ```python -m src.train```
	
	To train on colab: Upload the project folder to colab and run notebooks/train.ipynb

	Note: For expt_1, both local and colab training are fast. For expt_2, local is quite slow, colab is preferred.

4. Evaluate the flow matching model (locally)
	
	```python -m src.eval```
