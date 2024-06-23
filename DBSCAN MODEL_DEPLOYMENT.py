# Import Libraries
from flask import Flask, render_template, request
from sqlalchemy import create_engine
import pandas as pd

import pickle
model = pickle.load(open('dbscan_model.pkl', 'rb')) # DBScan clustering model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        user = request.form['user']
        pw = request.form['password']
        db = request.form['databasename']
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        try:

            data = pd.read_csv(f)
        except:
                try:
                    data = pd.read_excel(f)
                except:      
                    data = pd.DataFrame(f)
                  
        # Drop the unwanted features
        
        prediction = pd.DataFrame(model.fit_predict(data), columns = ['clusters'])
        prediction = pd.concat([data, prediction], axis = 1)
        
        prediction.to_sql('db_scan', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        html_table = prediction.to_html(classes='table table-striped')
        
        return render_template("data.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #e7e8bc;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")

if __name__=='__main__':
    app.run(debug = False)
