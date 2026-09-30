# README

To run this pilot code, create a python virtual environment using the `final_requirements.txt` file. 


Something like this should work

`python3 -m venv my_env`
`source my_env/bin/activate`
`pip install -r final_requirements.txt`

## Updated inscriptions

To run this code, you have to organize your h5 raw data file in a singular folder. The different cohorts should be located in different subfolders within a 'raw' folder. The file structure should look like :

MainDataFolder
- Group1
  - raw
- Group2
  - raw
 
For as many groups as you wish. 

In order to run the code, you should first process the data by typing `python main_processing_script.py` then analyse it with `python main_analysis_script`. All the code is in the src folder and should be kept organized as is. There are some parameters (path, framerate, and resolution) to adjust in the headers of both scripts. 


## Legacy instructions


To run the code, place the h5 files in the same folder as the file `inspect_data.py`.

Then run the code by typing `python inspect_data.py`. 

The code will save a series of figures in the *FiguresFolder* directory it will create. These contains the raw data of each individual file (there is an option to show these figures while running by changing line 19 from bool_plot=False to bool_plot=True).

For each data it will present two figures, (1) the raw position of the snout marker in the arena and (2) the position of all the differnet markers with time along the x-axis (color coded with legend). The most interesting figures are the second kind as they allow to assess whether the foot contact detection is possible (i.e. if we get clear step-like traces in for the individual paws, usually observable through zooming).


The last part of the code is just representing tentative at filtering the data wiht 6th order butterworth. 
