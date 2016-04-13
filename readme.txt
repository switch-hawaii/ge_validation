This contains files to run SWITCH and create similar results to the RPS Study conducted by HNEI using GE MAPS.

To install this repository:

1. install python, git and a solver (glpk, cplex, gurobi) (see the first half of https://github.com/switch-model/switch-hawaii-studies/blob/master/README.md (everything down to "install switch") for more information.)

3. execute the following commands in a terminal window (command prompt) to install this model and a matching copy of SWITCH:

cd <location to store model>
git clone --recursive https://github.com/switch-hawaii/ge_validation.git
cd ge_validation/switch
python setup.py develop
cd ..

4. After this, you can solve the model by cd'ing into the ge_validation directory and then running one of these commands:

switch solve
switch solve-scenarios

You can add --help to these commands to see more options.

5. Edit scenarios.txt, options.txt and modules.txt to configure the model as needed. You can also create new python modules in the local directory and then add their name to modules.txt. These modules should have similar functions to the files inside switch/switch_mod.
