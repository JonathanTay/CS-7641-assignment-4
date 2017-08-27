This folder contains the submission for assignment 4.

There are 2 subfolders: 
1) Solution - contains the python/jython code to run the assignment
2) BURLAP - Contains the Java BURLAP project and dependencies.

This project was done in Java 1.8.0.121, Jython 2.7.0, Python 3.5.2 (libraries include Pandas 0.18.0, numpy 1.11.3, matplotlib 1.5.1) and based on this repo: https://github.com/juanjose49/omscs-cs7641-machine-learning-assignment-4

To run the project, run the run.bat file in the Solution folder. This runs the jython code against the burlap.jar file.
To recompile the burlap.jar file, go to the BURLAP subfolder and run "ant dist", then copy the executable burlap.jar file to the Solution folder.
The plotter.py file plots most of the charts. The policy maps are available when running the code easyGW, hardGW and varysizeGW scripts in Jython.


Within the Solution folder, there are the outputs from the jython script. These are named: 
1) Easy <solver name>.csv - results on easy grid world
2) Hard <solver name>.csv - results on hard grid world
3) Size <solver name>.csv - results on size experiments
