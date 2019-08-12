A data exploration, analysis and modeling project on employee and applicant data to explore, analyze and predict potential length of applicants' employment at given organizations.

## Code Organization

+ data_analysis
    - data 
        - [file1].csv
        - [file2].csv
    - notebooks
        - data_analysis.ipynb
    - src
        - data_load_transform.py
        - visuals.py
        - model.py
    - .env
    - README.md
    - environment.yml
    
### Setup a conda environment

To get started, setup a new environment (preferably with anaconda) using  `./environment.yml` file

`conda env create -f path/to/environment.yml`

Set the following environment variables with the right value in your environment.

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


### Analysis

The analysis notebook can be found under `./notebooks` directory as `jupyter_data_analysis.ipynb`
Run the following command in your terminal from the `./notebooks` directory


```jupyter notebook jupyter_data_analysis.ipynb```


### Classifying Tenure Length

Here, we can train an algorithm of our choice and make predictions on our test data.
To tune a model, train, test and return a classification report, run the following command in `./src' directory from your terminal
 
 
```python model.py [[path to data directory | Default: ./data/]]```




