#imports.
from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import pickle

app = Flask(__name__)

#loading and reading the data.
model = pickle.load(open('C:\Machine-Learning-COM6033-Lab\lab7\code example\LinearRegressionModel.pkl', 'rb'))
data = pd.read_csv('C:\Machine-Learning-COM6033-Lab\lab7\code example\cleaned_car_data.csv')


#route to the page containing the form.
@app.route('/', methods=['POST', 'GET'])
def index():
    #loads all of the unique data in each column of the
    #DataFrame as options to select for the HTML form fields.
    name = sorted(data['name'].unique())
    #reversed so that dates appear from newest to oldest.
    year = sorted(data['year'].unique(), reverse=True)
    km_driven = sorted(data['km_driven'].unique())
    fuel = sorted(data['fuel'].unique())
    seller_type = sorted(data['seller_type'].unique())
    transmission = sorted(data['transmission'].unique())
    owner = sorted(data['owner'].unique())

    #if the form is submitted, do the following.
    if request.method == "POST":
        #retrieves inputted data from the form.
        form_name = request.form['name']
        form_year = request.form['year']
        form_fuel = request.form['fuel']
        form_seller_type = request.form['seller_type']
        form_transmission = request.form['transmission']
        form_owner = request.form['owner']
        #changes the option '3 or more' to '3' so 
        #it can be properly interpretted by the model.
        if form_owner=='3 or more':
            form_owner=3
        form_km_driven = request.form['km_driven']
        
        #prints the form data in the terminal to
        #check it has been correctly retrieved.
        print(form_name, form_year, form_fuel, 
        form_seller_type, form_transmission, 
        form_owner, form_km_driven)
        
        #redirects to the 'prediction' page and
        #passes through the form data.
        return redirect(url_for('prediction',
        form_name = form_name, 
        form_year = form_year,
        form_fuel = form_fuel,
        form_seller_type = form_seller_type,
        form_transmission = form_transmission,
        form_owner = form_owner,
        form_km_driven = form_km_driven))
    else:
        #renders the 'index.html' template and
        #passes through the unique values from
        #each column in the DataFrame.
        return render_template('index.html', 
        name = name,
        year = year,
        km_driven = km_driven,
        fuel = fuel,
        seller_type = seller_type,
        transmission = transmission,
        owner = owner)


#route to the page containing the prediction.
@app.route('/<form_name>/<form_year>/<form_fuel>/<form_seller_type>/<form_transmission>/<form_owner>/<form_km_driven>')
def prediction(form_name, form_year, form_fuel, form_seller_type, form_transmission, form_owner, form_km_driven):
    #storing the form data passed through from the 'index' page.
    name = form_name
    year = form_year
    fuel = form_fuel
    seller_type = form_seller_type
    transmission = form_transmission
    owner = form_owner
    km_driven = form_km_driven

    #makes a prediction using the regression model from my notebook.
    prediction = model.predict(pd.DataFrame([[name, year, fuel, seller_type, transmission, owner, km_driven]], 
    columns=['name', 'year', 'fuel', 'seller_type', 'transmission', 'owner', 'km_driven']))

    #reformats the prediction to 2 decimal places, because this value is given as Indian currency.
    prediction = '{:,.2f}'.format(prediction[0])
    #check for minus values and disregards them.
    if prediction[0] == '-':
        prediction = "The model can not accurately predict the price of this vehicle..."

    #renders the 'prediction.html' template, passes 
    #through the form data from the 'index' page,
    #and passes through the prediction value.
    return render_template('prediction.html', 
    form_name = form_name, 
    form_year = form_year,
    form_fuel = form_fuel,
    form_seller_type = form_seller_type,
    form_transmission = form_transmission,
    form_owner = form_owner,
    form_km_driven = form_km_driven,
    prediction = prediction)


if(__name__=="__main__"):
    app.run(debug=True)