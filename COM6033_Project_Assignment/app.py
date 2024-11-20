from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained models and scaler
scaler = joblib.load(r'C:\Machine-Learning-COM6033-Lab\COM6033_Project_Assignment\scaler.pkl')
rf_model = joblib.load(r'C:\Machine-Learning-COM6033-Lab\COM6033_Project_Assignment\rf_model.pkl')

# Load the dataset for feature extraction
df = pd.read_csv(r'C:\Machine-Learning-COM6033-Lab\COM6033_Project_Assignment\laptop_price.csv')
orig_df = df
df.drop(columns=['laptop_ID', 'Product', 'Inches'], inplace=True)  # Drop unnecessary columns
df.dropna(inplace=True)  # Drop rows with missing values

# Preprocess the memory column
def process_memory(memory):
    memory = memory.replace('GB', '').replace('TB', '000').replace('Hybrid', '').strip()
    parts = memory.split('+')
    storage = {"SSD": 0, "HDD": 0, "Flash_Storage": 0, "Hybrid": 0}
    
    for part in parts:
        part = part.strip()
        try:
            if 'SSD' in part:
                storage['SSD'] += int(float(part.replace('SSD', '').strip()))
            elif 'HDD' in part:
                storage['HDD'] += int(float(part.replace('HDD', '').strip()))
            elif 'Flash Storage' in part or 'Flash_Storage' in part:
                storage['Flash_Storage'] += int(float(part.replace('Flash Storage', '').replace('Flash_Storage', '').strip()))
            elif 'Hybrid' in part:
                storage['Hybrid'] += int(float(part.strip()))
        except ValueError:
            print(f"Unexpected format: {part}")
    
    return storage

# Preprocess the memory column to extract SSD, HDD, etc.
memory_data = df['Memory'].apply(process_memory)
df['SSD'] = memory_data.apply(lambda x: x['SSD'])
df['HDD'] = memory_data.apply(lambda x: x['HDD'])
df['Flash_Storage'] = memory_data.apply(lambda x: x['Flash_Storage'])
df['Hybrid'] = memory_data.apply(lambda x: x['Hybrid'])
df.drop(columns=['Memory'], inplace=True)

# Extract additional features and encode categorical variables
df['Cpu_Brand'] = df['Cpu'].apply(lambda x: x.split()[0])
df['Gpu_Brand'] = df['Gpu'].apply(lambda x: x.split()[0])
df['Resolution'] = df['ScreenResolution'].apply(lambda x: '4K' if '4K' in x else ('Full HD' if 'Full HD' in x else 'Others'))

# Use get_dummies for categorical columns, just like during model training
df = pd.get_dummies(df, columns=['Company', 'TypeName', 'OpSys', 'Resolution', 'Gpu_Brand'], drop_first=True)

# Split the dataset into features (X) and target (y)
X = df.drop(columns=['Price_in_euros'])
y = df['Price_in_euros']

# Standardize features
X = X.select_dtypes(include=[np.number])  # Ensure only numeric columns are used
X_train = scaler.fit_transform(X)

# Define the home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get data from the form
        form_data = {
            'name': request.form['name'],
            'ram': int(request.form['ram']),
            'cpu_brand': request.form['cpu_brand'],
            'gpu_brand': request.form['gpu_brand'],
            'resolution': request.form['resolution'],
            'weight': float(request.form['weight']),
            'ssd': int(request.form['ssd']),
            'hdd': int(request.form['hdd']),
            'flash_storage': int(request.form['flash_storage']),
            'hybrid': int(request.form['hybrid'])
        }
        
        # Convert the form data to DataFrame
        user_data = pd.DataFrame([form_data])

        # Perform the same preprocessing as done for the training data:
        
        # 1. Process memory
        user_data['SSD'] = form_data['ssd']
        user_data['HDD'] = form_data['hdd']
        user_data['Flash_Storage'] = form_data['flash_storage']
        user_data['Hybrid'] = form_data['hybrid']
        
        # 2. Handle categorical variables (One-hot encode)
        user_data = pd.get_dummies(user_data, columns=['cpu_brand', 'gpu_brand', 'resolution'], drop_first=True)
        
        # 3. Align user_data columns to match the training data
        # Add missing columns and fill with zeros (important for one-hot encoding mismatches)
        user_data = user_data.reindex(columns=X.columns, fill_value=0)
        
        # 4. Apply the same scaling
        user_data_scaled = scaler.transform(user_data)

        # Make the prediction
        prediction = rf_model.predict(user_data_scaled)
        return render_template('prediction.html', prediction=prediction[0])
    
    return render_template('index.html')

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
