# patchclamp

Jupyter Notebooks for data analysis of the voltage-sensing patch-clamp experiment.

*[Notebooks Index](http://nbviewer.ipython.org/github/tritemio/voltagesensing/tree/master/)*

- [Patch Clamp Analysis - Phase offset-take1](http://nbviewer.ipython.org/github/tritemio/voltagesensing/blob/master/Patch%20Clamp%20Analysis%20-%20Phase%20offset-take1.ipynb)
- [Patch Clamp Analysis - Phase offset-take2](http://nbviewer.ipython.org/github/tritemio/voltagesensing/blob/master/Patch%20Clamp%20Analysis%20-%20Phase%20offset-take2.ipynb)
- [Patch Clamp Analysis - Phase offset-take3](http://nbviewer.ipython.org/github/tritemio/voltagesensing/blob/master/Patch%20Clamp%20Analysis%20-%20Phase%20offset-take3.ipynb)
- [Patch Clamp Analysis - Phase offset-fov1](http://nbviewer.ipython.org/github/tritemio/voltagesensing/blob/master/Patch%20Clamp%20Analysis%20-%20Phase%20offset-fov1.ipynb)
- [Patch Clamp Analysis - FFT](http://nbviewer.ipython.org/github/tritemio/voltagesensing/blob/master/Patch%20Clamp%20Analysis%20-%20FFT.ipynb)


# Requirements

> Users with no experience with python and the jupyter notebooks should read:
> 
> - [Jupyter/IPython Notebook Quick Start Guide](http://jupyter-notebook-beginner-guide.readthedocs.org/)


Running these notebooks requires python and a small set of scientific libraries. 
The detailed list of dependencies can be found in [`environment.yml`](https://github.com/tritemio/voltagesensing/blob/master/environment.yml).

To automatically re-create the computational environment used here please
install the scientific python distribution Continuum Ananconda and follow 
the instructions below.

Anaconda will install the latest version of all the major scientific python libraries.
For long term reproducibility, you can re-create an enviroment containing the exact version 
of each libray used in the present analysis by typing the following command in a terminal
(*cmd.exe* on Windows, *Terminal* app on OSX):

```
conda env create --name=voltagesensing_env --file=environment.yml
```

The environment must be activated to be used. On windows type in a terminal:

```
activate voltagesensing_env
```

or with (Linux, Mac OSX):

```
source activate voltagesensing_env
```

Now, in the same terminal use for the previous command, launch the Jupyter/IPython Notebook with:

```
ipython notebook
```

This last command will open a browser showing the Jupyter Notebook dashboard 
(recommended browser Firefox or Chrome).
From the dashboard enter the folder containing a copy of the present repository, 
then click on one of the provided notebooks.

If it's the first time you open a notebook, we suggest to take a quick tour of the interface 
by clicking on the menu *Help* -> *User Interface Tour*. 
Further information on the *Jupyter Notebook Application* can be found at http://ipython.org/notebook.html.
