# Forecast use of a city bikeshare system

It is a kaggle competition in which participants are asked to combine historical usage patterns with weather data in order to forecast bike rental demand in the Capital Bikeshare program in Washington, D.C.

The data used for competition is present in `data` folder. The `code` folder contains `ensemble.py`, an ensemble of Random Forest and Gradient Boosting. It can be run as follows

    cd code
    python ensemble.py

The predictions made by ensemble.py achieve a score of 0.36996 on kaggle.
