# PAM_P3S

## Install

To run our scripts, you will need to have a specific environment which will require the installation of miniconda (or anaconda). 
If you do not already have it, you can install it from the original [website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).


- Clone the github repository

``` 
git clone https://github.com/PierreChouteau/PAM_P3S.git
``` 

- Create a virtual env with python 3.10.8:

``` 
conda create -n P3S python=3.10.8
``` 

- Activate the environment:
``` 
conda activate P3S
``` 

- Go into the repository and install the dependencies: 
``` 
cd PAM_P3S
pip install -r requirements.txt
``` 