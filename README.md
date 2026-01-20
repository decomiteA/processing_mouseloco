# README

To run this pilot code, create a python virtual environement using the `requirements.txt` file. 


Something like this should work

`python3 -m venv my_env`
`source my_env/bin/activate`
`pip install -r requirements.txt`


To run the code, place the h5 files in the same folder as the file `inspect_data.py`.

Then run the code by typing `python inspect_data.py`. 

The code will save a series of figures in the *FiguresFolder* directory it will create. These contains the raw data of each individual file (there is an option to show these figures while running by changing line 19 from bool_plot=False to bool_plot=True).

For each data it will present two figures, (1) the raw position of the snout marker in the arena and (2) the position of all the differnet markers with time along the x-axis (color coded with legend). The most interesting figures are the second kind as they allow to assess whether the foot contact detection is possible (i.e. if we get clear step-like traces in for the individual paws, usually observable through zooming).


The last part of the code is just representing tentative at filtering the data wiht 6th order butterworth. 
