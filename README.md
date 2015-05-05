# patchclamp

Notebooks for data analysis of the voltage-sensing patch-clamp experiment.

*[Notebooks Index](http://nbviewer.ipython.org/github/tritemio/voltagesensing/tree/master/)*

- [Patch Clamp Analysis - Phase offset-take1](http://nbviewer.ipython.org/github/tritemio/voltagesensing/blob/master/Patch%20Clamp%20Analysis%20-%20Phase%20offset-take1.ipynb)
- [Patch Clamp Analysis - Phase offset-take2](http://nbviewer.ipython.org/github/tritemio/voltagesensing/blob/master/Patch%20Clamp%20Analysis%20-%20Phase%20offset-take2.ipynb)
- [Patch Clamp Analysis - Phase offset-take3](http://nbviewer.ipython.org/github/tritemio/voltagesensing/blob/master/Patch%20Clamp%20Analysis%20-%20Phase%20offset-take3.ipynb)
- [Patch Clamp Analysis - Phase offset-fov1](http://nbviewer.ipython.org/github/tritemio/voltagesensing/blob/master/Patch%20Clamp%20Analysis%20-%20Phase%20offset-fov1.ipynb)
- [Patch Clamp Analysis - FFT](http://nbviewer.ipython.org/github/tritemio/voltagesensing/blob/master/Patch%20Clamp%20Analysis%20-%20FFT.ipynb)

# Requirements

Running this notebooks requires python and a small set of scientific libraries. 
The detailed list of dependencies can be found in [`environment.yml`](https://github.com/tritemio/voltagesensing/blob/master/environment.yml).

To automatically re-create the computational environment used here please
install the scientific python distribution Continuum Ananconda and follow 
the instructions below.

Once `conda` is installed (included in Anaconda), recreate an environment with 
all the dependencies (and their exact versions):

```
conda env create --name=voltagesensing_env --file=environment.yml
```

The new environment can be activate with (Windows):

```
activate voltagesensing_env
```

or with (Linux, Mac OSX):

```
source activate voltagesensing_env
```




