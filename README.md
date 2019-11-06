# Gadolinium prediction

This is the pythoncode that is similar to the code that was used in the paper [Can Virtual Contrast Enhancement in Brain MRI Replace Gadolinium?
A Feasibility Study](https://journals.lww.com/investigativeradiology/Abstract/publishahead/Can_Virtual_Contrast_Enhancement_in_Brain_MRI.98899.aspx)


## Installation
In order to install the package, clone this repository by executing the following command in the terminal:

```
git clone https://github.com/nikolasmorshuis/gadolinium_prediction.git
```

Change into this directory using
```
cd gadolinium_prediction
```

Create a new virtual environment with pip:

```
python3 -m venv env
```

Activate the environment:

```
source env/bin/activate
```

Install the requirements from the requirements.txt file:

```
pip install -r requirements.txt
```

Navigate to the second folder gadolinium_prediction and add the current path to your PYTHONPATH:

```
cd gadolinium_prediction
export PYTHONPATH=$(pwd)
```

The MRI-data is typically large 3D-data. I wrote a tool to extract Patches of adjustable size from the 3D-data that is available here: https://github.com/nikolasmorshuis/PatchCreator . Because the MRI-files are too large to be predicted at once by our network, the PatchCreator-repository is used here as well. You can clone and install it using the link.

After the above steps are performed, the scripts can be used. 

## Folder structure and where to put your data

The folder structure is as follows:  
 \\data  
 \_\\train  
 ...\_\\Patient_1  
 ......\_\\..._T1.nii.gz  
 ......\_\\..._T1_CE.nii.gz  
 ......\_\\...
 
You can distribute your data in the train, validation and test folder as you like.

Note that each patient should have his/her own folder, that contains all the necessary data.

In the config.yaml file, you can specify the type of sequences that you like to use. Note that the names of the sequences should end according to the file that you wrote. If e.g. T2_flair is written down config.yaml file, then all T2_flair files should end like: _T2_flair.nii.gz .

Note that the program expects to get a mask of the brain region as input. The mask-files should have the ending _mask.nii.gz .

## How to use the scripts

To preprocess the data, execute the script that can be found in the preprocessing directory:

```
python preprocessing/data_creator.py
```

After the data is preprocessed, you can start to train the neural network:

```
python train.py
```

The resulting models are saved in the directory data_output by default.
