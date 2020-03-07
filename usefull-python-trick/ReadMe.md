# SETUP
## Step 1 : Setup Anaconda
	While installing Anaconda make sure that you check both options:
	a. Add Anaconda to my PATH environment variable
	b. Register Anaconda as my default Python

## Step 2 : Create Virtual Environment
	Open the Anaconda command prompt and execute the following command.
	`conda create --name opencv-env python=3.6`

## Step 3 : Install OpenCV
	3.1. Activate the environment: 
		`activate opencv-env`
	3.2. Install OpenCV and other important packages
	Continuing from the above prompt, execute the following commands
		`pip install numpy scipy matplotlib scikit-learn jupyter`
		`pip install opencv-contrib-python`
		`pip install dlib`
	3.3. Test your installation
	Open the python prompt on the command line by typing python on the command prompt
		`import cv2`
		`cv2.__version__`
		`import dlib`
		`dlib.__version__`

