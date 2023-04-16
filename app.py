import gradio as gr
import numpy as np
import pandas as pd
import pickle
import sklearn


#load the model toolkit
with open("model_toolkit.pkl", "rb") as f:
    app_toolkit = pickle.load(f)


#load all the app key components
num_imputer = app_toolkit["numerical_imputer"]
cat_imputer = app_toolkit["categorical_imputer"]
encoder = app_toolkit["encoder"]
scaler = app_toolkit["scaler"]
model = app_toolkit["Final_model"]


#define a function to input variables for prediction
def predict(Gender,SeniorCitizen,Partner,Dependents, Tenure, PhoneService,MultipleLines,
                       InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,
                       Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges):

    #create a dataframe from the variables inputed
    inputvr_df = pd.DataFrame({
        'gender': [Gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [Tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
        })

    # separate the categorical and numerical variables
    cat_columns = [col for col in inputvr_df.columns if inputvr_df[col].dtype == 'object']
    num_columns = [col for col in inputvr_df.columns if inputvr_df[col].dtype != 'object']

    #apply the imputer on the separeted variables
    impute_cat = cat_imputer.transform(inputvr_df[cat_columns])
    impute_num = num_imputer.transform(inputvr_df[num_columns])

    #apply the encoder on categorical variables
    encoded_cat = pd.DataFrame(encoder.transform(impute_cat).toarray(),
                                   columns=encoder.get_feature_names_out(cat_columns))
    
    #apply the scaler on the numerical variables
    num_scaled = scaler.transform(impute_num)
    # turn the scaled variables into a dataframe
    num_df = pd.DataFrame(num_scaled , columns = num_columns)

    #join both dataframes
    joined_df = pd.concat([encoded_cat, num_df], axis=1)


    # Make a prediction
    prediction = model.predict(joined_df)

    predict_outcome = "Customer is likely to churn" if prediction[0] == "Yes" else "Customer is not likely to churn"


input_inf=[]
#build the interface
with gr.Blocks() as churn:

    Title= gr.Label("PREDICTING CUSTOMER CHURN IS A TELCO COMPANY")
    with gr.Row():
        Title
    
    with gr.Row():    
      gr.Markdown(
        """
        ### Kindly fill the all the customers details in the provided space to help predict Churn
     """   )
    
    with gr.Row():
        input_inf = [
                gr.components.Radio(['male', 'female'], label='Select gender'),
                gr.components.Number(label="A Seniorcitizen?; No=0 and Yes=1"),
                gr.components.Radio(['Yes', 'No'], label='Has Partner?'),
                gr.components.Dropdown(['No', 'Yes'], label='Any Dependents? '),
                gr.components.Number(label='Duration with Telco)'),
                gr.components.Radio(['No', 'Yes'], label='PhoneService Subscribed? '),
                gr.components.Radio(['No', 'Yes'], label='MultipleLines Subscribed?'),
                gr.components.Radio(['DSL', 'Fiber optic', 'No'], label='InternetService Subscribed?'),
                gr.components.Radio(['No', 'Yes'], label='OnlineSecurity Subscribed?'),
                gr.components.Radio(['No', 'Yes'], label='OnlineBackup Subscribed?'),
                gr.components.Radio(['No', 'Yes'], label='DeviceProtection Subscribed?'),
                gr.components.Radio(['No', 'Yes'], label='TechSupport Subscribed?'),
                gr.components.Radio(['No', 'Yes'], label='StreamingTV Subscribed?'),
                gr.components.Radio(['No', 'Yes'], label='StreamingMovies Subscribed?'),
                gr.components.Dropdown(['Month-to-month', 'One year', 'Two year'], label='which Contract in use?'),
                gr.components.Radio(['Yes', 'No'], label='Do you prefer PaperlessBilling?'),
                gr.components.Dropdown(['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                        'Credit card (automatic)'], label='Which PaymentMethod used?'),
                gr.components.Number(label="Enter monthly charges"),
                gr.components.Number(label="Enter total charges")
            ]
    with gr.Row():

       
       predict_button = gr.Button("Predict").style(full_width=True)

    #output function
    output_inf = gr.Label(label = "Churn")

    predict_button.click(fn=predict, inputs=input_inf, outputs=output_inf) 



churn.launch(share=True, server_port=8080)


                                                                               
    
