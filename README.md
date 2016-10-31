# arena_analysis
A Data Analysis and modeling challenge for arena.io

## Code Organization

+ arena_analysis
    - data 
        - [file1].csv
        - [file2].csv
    - notebooks
        - arena_analysis.ipynb
    - src
        - data_load_transform.py
        - visuals.py
        - model.py
    - .env
    - README.md
    - environment.yml
    
### Setup a conda environment

To get started, setup a new environment (preferably using anaconda) using  `./environment.yml` file

`conda env create -f path/to/environment.yml`

Set the following environment variables with the right value
The default values are as follows:
```
APPLICANTS_DATA=data_exercise_applicants.csv
HIRES_DATA=data_exercise_hires.csv
DATA_DIR=../data
REPORT_DIR=../reports
```

### Data Munging and preprocessing

Data munging is done by running the data_load_transform.py module.

To run the data munging module, go into the `./src` subdirectory and type the following in your terminal

``` python data_load_transform.py ```


This will, 
- load the datasets (applicants and hires)
- preprocess the data (encoding target, binarizing categorical variables, filling nan values, etc)
- splits the data into test and train samples


### Data Analysis

The analysis notebook can be found under `./notebooks` directory as `arena_data_analysis.ipynb`
Run the following command in your terminal from the `./notebooks` directory


```ipython notebook arena_data_analysis.ipynb```


### Predictive Modeling

To tune a model, train, test and return a classification report, run the following command in `./src' directory from your terminal
 
 
```python model.py```




