# >>>>>>>>>>>>>>>>> Importing the packeges & setting the constants <<<<<<<<<<<<<<
# import keras
import joblib
import sklearn

import polars as pl
import pandas as pd
import numpy as np
from shiny import App, ui, render

COLORS  = ['#102542', '#F87060', '#CDD7D6', '#B3A394', '#FFFFFF']
COLUMNS = ['Gender', 'Married', 'Education',
           'Property_Area', 'Monthly_Income',
           'Extra_Monthly_Income', 'Loan_Term',
           'Credit_History', 'Loan_Status',
           'Dependents', 'Employment_Type',
           'Loan_Amount', 'Total_Monthly_Income']


df = pd.read_csv('data-cleaning/cleaned-data/processsed-data.csv')
# ===============================================================================


# >>>>>>>>>>>>>>>>>>>> Loading the models & data preparing <<<<<<<<<<<<<<<<<<<<<<
scaler               = joblib.load('models/scaler.joblib')
encoder              = joblib.load('models/encoder.joblib')
lr_dt_model          = joblib.load('models/lr_dt.joblib')
# nn_model             = keras.models.load_model('models/nn-model.h5')

def preprocess_input(arr: np.array):
    
    '''This function prepare the data so the model can take 
    an input and give us the predicted value.'''
    
    df = pd.DataFrame([arr.tolist()], columns= COLUMNS)
    
    df = df.astype({'Loan_Status'   : 'category', 'Credit_History'  : 'category',
                    'Property_Area' : 'category', 'Employment_Type' : 'category',
                    'Married'       : 'category', 'Education'       : 'category',
                    'Gender'        : 'category', 'Monthly_Income'  : np.uint16,
                    'Loan_Amount'   : np.uint16,  'Loan_Term'       : np.uint16,
                    'Dependents'    : np.uint16, 
                    'Total_Monthly_Income' : np.uint16,
                    'Extra_Monthly_Income' : np.uint16})


    continuous_cols = df.select_dtypes(include=np.number).columns
    continuous_data = df[continuous_cols]
    continuous_data[continuous_cols] = scaler.transform(continuous_data)

    categorical_cols = df.select_dtypes('category').columns.drop(['Loan_Status'])
    categorical_data = encoder.transform(df[categorical_cols])
    categorical_cols = encoder.get_feature_names_out(categorical_cols)
    categorical_data = pd.DataFrame(categorical_data,
                                       columns=[col for col in categorical_cols])

    processsed_arr   = pd.concat([continuous_data, categorical_data], axis= 1)

    return processsed_arr
# ==============================================================================


# >>>>>>>>>>>>>>>>>>>>>>>>> Shiny app Inputs & Styling <<<<<<<<<<<<<<<<<<<<<<<<<
with open('app/style.css', 'r') as file:
    css_code = file.read()

app_ui = ui.page_fluid(
    ui.tags.style(css_code),
    
    ui.panel_title(title= '', window_title= 'Loan prediction ML app'),
    
    ui.div({'class': 'container'},

        ui.h1('Loan Acceptance Prediction ML App'),
        ui.p('*Don\'t use thie ML app in real world it was trained on mock data.'),
           
        ui.h5({'class': 'header'}, 'Numrical Inputs'),
        ui.div({'class': 'horizontal-container'},
            
            ui.div(
                {'class': 'card'},
                ui.input_slider('term', 'How long is your Loan term in days?',
                min=30, max=500, value=200, post= ' Days')),

            ui.div(
                {'class': 'card'},
                ui.input_slider('amount', 'What\'s your requested loan amount?:',
                min=5, max=100, value=40, post= 'K')),
                
            ui.div(
                {'class': 'card'},
                ui.input_numeric('extra_income', 'What\'s your extra monthly income if you have?:',
                min=0, max=1_000_000, value=1000)),
                
            ui.div(
                {'class': 'card'},
                ui.input_numeric('income', 'How much money do you get per month?:',
                min= 500, max= 1_000_000, value= 2000)),
                
            ui.div(
                {'class': 'card'},
                ui.input_slider('history', 'What\'s your credit history?:',
                min=300, max=850, value=670))
        ),

        ui.h5({'class': 'header'}, 'Categorical Inputs'),
        ui.div({'class': 'horizontal-container'},
        
            ui.div({'class': 'card'},
                ui.input_select('area', 'Choose your desired property area',
                ['Urban', 'Semiurban', 'Rural'])),
            
            ui.div({'class': 'card'},
                ui.input_select('sex', 'What\'s your Gender?',
                ['Female', 'Male'])),
                
            ui.div({'class': 'card'},
                ui.input_select('self_employed', 'Are you self employed?',
                ['Yes', 'No'])),
                
            ui.div({'class': 'card'},
                ui.input_select('married', 'Are you married?',
                ['Yes', 'No'])),
                
            ui.div({'class': 'card'},
                ui.input_select('education', 'Did you graduate from colleuge?', 
                ['Graduate', 'Not Graduate'])),
                
            ui.div({'class': 'card'},
                ui.input_select('dependents', 'How many dependents do you have?', 
                ['0', '1', '2', '3+']))
            ),
            
        ui.h5({'class': 'header'}, 'The Loan Will Be:'),
        ui.div({'class': 'output-container'},
          ui.output_text("predict_loan_status"))
    )
)

def server(input, output, session):

    @output
    @render.text
    def predict_loan_status():
      
        gender               = input.sex()
        married              = input.married()
        education            = input.education()
        property_area        = input.area()
        monthly_income       = input.income()
        extra_monthly_income = input.extra_income()
        loan_term            = input.term()
        credit_history       = input.history()
        credit_history       = 'Bad' if credit_history < 670 else 'Good'
        loan_Status          = 'Rejected' # Just a temp value
        dependents           = input.dependents()
        dependents           = 3 if dependents == '3+' else int(dependents)
        self_employed        = input.self_employed()
        self_employed        = 'Self-Employed' if self_employed == 'Yes' else 'Not-Self-Employed'
        loan_amount          = input.amount() * 1000
        total_income         = extra_monthly_income + monthly_income


        unprocessed_arr      = np.array([gender,        married,
                                        education,      property_area,
                                        monthly_income, extra_monthly_income,
                                        loan_term,      credit_history,
                                        loan_Status,    dependents,
                                        self_employed,  loan_amount,
                                        total_income])

        processed_arr        = preprocess_input(unprocessed_arr)
        loan_status          = 'Accepted' if lr_dt_model.predict(processed_arr)[0] else 'Rejected'
        
        if total_income < 500:
            loan_status = 'Please don\'t input total income less than 500$'

        return loan_status


app = App(app_ui, server)
# ==============================================================================
