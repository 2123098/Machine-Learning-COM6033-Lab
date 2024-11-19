# Car Price Predictor

This project is a web app built with Python, the Flask microframework, and a machine learning regression model. I used a car data set and a linear regression algorithm to train the model and, as the name suggests, the web app uses this model to predict the price of different cars.

+ All of the python code is in the `app.py` file.
+ The 2 HTML page templates can be found in the `templates` directory.
+ My CSS style rules can be found in the `static\css` directory in the `style.css` file.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running this application, you need to download the following things:
+ A code editor; I used [Visual Studio Code](https://code.visualstudio.com/download)
+ [Python version 3.7.0](https://www.python.org/ftp/python/3.7.0/python-3.7.0.exe), or visit https://www.python.org/downloads/release/python-370/ to select a specific installer.

Click on each of the above for the corresponding download link.

You will also need a cloned version of this repository on your machine. 

### Installing

To build and run the application, you will first need to create a Python virtual environment so that all the project files are isolated from the rest of your machine - to prevent conflicts or unwanted interactions with the rest of your system.

+ From Windows Powershell use the `cd` command to move into the `6033-software-artefact` directory and run:

```
virtualenv venv -p "path\to\python3.7.0\executable\file"
```

+ The command looked like this for my machine:

```
virtualenv venv -p "C:\Users\UserName\AppData\Local\Programs\Python\Python37\python.exe"
```

+ However, you will need to replace the path (after `-p`) with the path to the python 3.7.0 executable file on your own machine.

If this command runs correctly, you will have a virtual environment directory (`venv`) in the `6033-software-artefact` directory.

To activate this virtual environment you will need to run:

```
venv\Scripts\activate
```

+ After running this command you should see a `(venv)` to the left of your current file location as seen below:

```
(venv) PS C:\Users\UserName\Documents\University\CompSci\Year3\COM6033\6033-software-artefact>
```

+ This indicates that the virtual environment has been successfully activated.

+ To install the other dependencies for this project, listed in the `requirements.txt` file, you will need to run:

```
pip install -r requirements.txt
```

+ After running this you will have installed all of the project dependencies. So, to the run the flask web application you need to run:

```
flask run
```

+ Now you will be able to visit the app in your browser using the url: [localhost:5000](http://localhost:5000) or http://127.0.0.1:5000/.

## Making and Viewing Changes

You can make changes to the source code but you will need quit the current deployment, using `CTRL+C`, and rerun the app using `flask run` to see the changes you made. 

You can also make changes to the model by modifying the Jupyter Notebook file (`Software_Artefact.ipynb`). Uploading this file to [Google Colab](https://colab.research.google.com/), and saving it using your Google account, will allow you to make changes in a pre-built browser environment with pre-installed machine learning libraries. However, during modification you must pay close attention because changing the model might break some of the functionality in the `app.py`, which will cause errors when attempting to run the web app. New data can be added to the model by adding it to the Jupyter Notebook file, but it must be provided in the CSV format. This data must also be cleaned and concatenated with the other cleaned data to form a single CSV file called `cleaned_car_data.csv` - refer to the steps in my Jupyter Notebook for guidance on this. If you make changes to the model you will need to re-export it to access these changes. You can do this by using the `pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))` command - found at the end of the notebook. You will then need to download this new `LinearRegressionModel.pkl` file and replace the old one with the new.


## Built With

+ **Windows 11** - The operating system used
+ **HTML** - The markup language for my front-end
+ **CSS** - The stylesheet for my front-end
+ **Python (version 3.7.0)** - The main programming language
+ **Flask (version 2.2.2)** - The web microframework used
+ **scikit-learn (version 1.0.2) and pandas (version 1.1.5)** - The two main python libraries I used

## Authors

+ **Student Name** - [Leeds Trinity University](https://www.leedstrinity.ac.uk/)