This is a sample of the 3DPeople Dataset with 1 sequence of 1 person.

To visualize the sample data:
```
conda create -n 3dpeople
source activate 3dpeople
conda install matplotlib opencv pillow scipy
conda install -c conda-forge ipywidgets=7.2.1
conda install -c plotly chart-studio
jupyter nbextension enable --py widgetsnbextension
jupyter notebook vis_dataset.ipynb 
```

website: https://cv.iri.upc-csic.es/
