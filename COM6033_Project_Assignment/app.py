from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

#Initialising Flask app
app = Flask(__name__)

#Loading the trained Linear Regression model and dataset
model = joblib.load(r'C:\Machine-Learning-COM6033-Lab\COM6033_Project_Assignment\LinearRegressionModel.pkl')
data = pd.read_csv(r'C:\Machine-Learning-COM6033-Lab\COM6033_Project_Assignment\cleaned_laptop_data.csv')
scaler = joblib.load(r'C:\Machine-Learning-COM6033-Lab\COM6033_Project_Assignment\scaler.pkl')
feature_names = data.drop(columns=['Price_in_euros']).columns 

#Loading the dataset for form input
input_data_for_dropdown = pd.read_csv(r'C:\Machine-Learning-COM6033-Lab\COM6033_Project_Assignment\laptop_price.csv')

#Extracting unique values for dropdown fields
ram = sorted(data['Ram'].unique())
cpu_brands = sorted(data['Cpu_Brand'].unique())
weights = sorted(data['Weight'].unique())
companies = sorted(data['Company'].unique())
typenames = [col.replace('TypeName_', '') for col in data.columns if col.startswith('TypeName_')]
opsystems = [col.replace('OpSys_', '') for col in data.columns if col.startswith('OpSys_')]
resolutions = [col.replace('Resolution_', '') for col in data.columns if col.startswith('Resolution_')]
gpu_brands = [col.replace('Gpu_Brand_', '') for col in data.columns if col.startswith('Gpu_Brand_')]

# Route for the form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #Retrieving form data
        form_ram = int(request.form['ram'])
        form_cpu_brand = request.form['cpu_brand']
        form_weight = float(request.form['weight'])
        form_company = request.form['company']
        form_typename = request.form['typename']
        form_opsystem = request.form['opsystem']
        form_resolution = request.form['resolution']
        form_gpu_brand = request.form['gpu_brand']
        form_ssd = int(request.form['ssd'])
        form_hdd = int(request.form['hdd'])
        form_flash_storage = int(request.form['flash_storage'])
        form_hybrid = int(request.form['hybrid'])

        #Preparing typename one-hot encoded data
        typename_columns = [col for col in data.columns if col.startswith('TypeName_')]
        typename_data = {col: 0 for col in typename_columns}
        typename_column = f'TypeName_{form_typename}'
        if typename_column in typename_data:
            typename_data[typename_column] = 1

        #Same for resolution one-hot encoded data
        resolution_columns = [col for col in data.columns if col.startswith('Resolution_')]
        resolution_data = {col: 0 for col in resolution_columns}
        resolution_column = f'Resolution_{form_resolution}'
        if resolution_column in resolution_data:
            resolution_data[resolution_column] = 1

        #Finally, for GPU brand one-hot encoded data
        gpu_columns = [col for col in data.columns if col.startswith('Gpu_Brand_')]
        gpu_data = {col: 0 for col in gpu_columns}
        gpu_column = f'Gpu_Brand_{form_gpu_brand}'
        if gpu_column in gpu_data:
            gpu_data[gpu_column] = 1

        #Handling one-hot encoding for OpSys
        opsys_columns = [col for col in data.columns if col.startswith('OpSys_')]
        opsys_data = {col: 0 for col in opsys_columns}
        opsys_column = f'OpSys_{form_opsystem}'
        if opsys_column in opsys_data:
            opsys_data[opsys_column] = 1

        #Creating input dataframe for prediction
        input_data = pd.DataFrame([{
            'Ram': form_ram,
            'Weight': form_weight,
            'SSD': form_ssd,
            'HDD': form_hdd,
            'Flash_Storage': form_flash_storage,
            'Hybrid': form_hybrid,
            'Cpu_Brand': form_cpu_brand,
            'Company': form_company,
            **typename_data,       
            **resolution_data,
            **gpu_data,
            **opsys_data
        }])

        #One-hot encoding for other categorical fields (Cpu_Brand, Company)
        input_data = pd.get_dummies(input_data, columns=['Cpu_Brand', 'Company'])

        #Aligning input data with feature names (in case some columns were missing)
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        #Making prediction
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        #Redirecting to prediction result
        return redirect(url_for('prediction', predicted_price=prediction))
    
    #Render the form
    return render_template('index.html', ram=ram, cpu_brands=cpu_brands, weights=weights,
                           companies=companies, typenames=typenames, opsystems=opsystems,
                           resolutions=resolutions, gpu_brands=gpu_brands)

#Route for prediction result
@app.route('/prediction')
def prediction():
    predicted_price = request.args.get('predicted_price')
    return render_template('prediction.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
