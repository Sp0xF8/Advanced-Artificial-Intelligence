# %%
#imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import shap 
import lime
import torch
from sentence_transformers import SentenceTransformer as st
import sklearn as sk
import imblearn



# %% [markdown]
# # Globals
# 

# %%
# load the data
base_claims_df = pd.read_csv('data/anno_claims.csv')
pd.set_option('display.max_columns', None)

# load the sentence transformer model
st_model = st("all-MiniLM-L6-v2")

Yes_No_Columns = ['Exceptional_Circumstances', 'Minor_Psychological_Injury', 'Whiplash', 'Police Report Filed', 'Witness Present']
Categorical_Columns = ['AccidentType', 'Dominant injury', 'Vehicle Type', 'Weather Conditions', 'Accident Description', 'Injury Description']


test_headder = 'SettlementValue'

impution_dic = { ## used to store which header needs to be imputed and which do not.
    'needed': [
        'AccidentType',
        'GeneralFixed',
        'Minor_Psychological_Injury',
        'Dominant injury',
        'Whiplash',
        'Weather Conditions',
        'Accident Description',
    ],
    'not_needed': [ ## these are columns which are no input could be used as a form of imput; meaning they are not applicable to the situation and were left blank for a reason.
        'Injury_Prognosis', ## should be defaulted to Z. 0 months
        'SpecialHealthExpenses', ## should be defaulted to 0.0 as not everyone has health expenses
        'SpecialReduction', ## should be defaulted to 0.0 as not everyone has reduction
        'SpecialOverage', ## should be defaulted to 0.0 as not everyone has overage
        'GeneralRest', ## should be defaulted to 0.0 as not everyone has rest
        'SpecialAdditionalInjury', ## should be defaulted to 0.0 as not everyone has additional injury
        'SpecialEarningsLoss', ## should be defaulted to 0.0 as not everyone has earnings loss
        'SpecialUsageLoss', ## should be defaulted to 0.0 as not everyone has usage loss
        'SpecialMedications', ## should be defaulted to 0.0 as not everyone has medications
        'SpecialAssetDamage', ## should be defaulted to 0.0 as not everyone has asset damage
        'SpecialRehabilitation', ## should be defaulted to 0.0 as not everyone has rehabilitation
        'SpecialFixes', ## should be defaulted to 0.0 as not everyone has fixes
        'GeneralUplift', ## should be defaulted to 0.0 as not everyone has uplift
        'SpecialLoanerVehicle', ## should be defaulted to 0.0 as not everyone has loaner vehicle
        'SpecialTripCosts', ## should be defaulted to 0.0 as not everyone has trip costs
        'SpecialJourneyExpenses', ## should be defaulted to 0.0 as not everyone has journey expenses
        'SpecialTherapy', ## should be defaulted to 0.0 as not everyone has therapy
        'Exceptional_Circumstances', ## should be defaulted to False as not everyone has exceptional circumstances
        'Number of Passengers', ## should be defaulted to 0 as not everyone has passengers
        'Injury Description', ## No input could be used as a form of imput; meaning they are not applicable to the situation and were left blank for a reason.
        'Police Report Filed', ## No input could be used as a form of imput; meaning they are not applicable to the situation and were left blank for a reason.
        'Witness Present', ## No input could be used as a form of imput; meaning they are not applicable to the situation and were left blank for a reason
    ],
    'drop': { ## these are columns which should be droppped if they are null as their values are highly correlated with the settlement value
        'SettlementValue',
        'Vehicle Age',
        'Vehicle Type',
        'Driver Age',
        'Accident Date',
        'Claim Date',
    }
}

# %% [markdown]
# # Creating Helper Functions
# These are useful for easily tranforming a sentence or column into embedded strings, capturing meaning behind words instead of just using a numerical value for the "category".

# %%

def encode_text(text):
    return st_model.encode(text, convert_to_tensor=True)


def transform_columns_to_embeddings(df, column_name):

    ids_to_string = {} # integer: string
    string_ids = {} # string: integer

    unique_column_values = df[column_name].unique()
    for index, value in enumerate(unique_column_values):
        if isinstance(value, str):
            string_ids[value] = index
            ids_to_string[index] = value
        else:
            print(f"Skipping non-string value: {value}")


    # Check if the column exists in the DataFrame
    embeddings = {} # hash(value): [embedding]
    strings = {} # hash(value): string
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Iterate through the DataFrame and encode the specified column
    unique_column_values = df[column_name].unique()
    for value in unique_column_values:
        if isinstance(value, str):  # Ensure the value is a string before encoding
            embedding = encode_text(value)

            ## convert embedding to an array of floats
            embedding = np.array(embedding.cpu()).astype(np.float32).tolist()

            embeddings[hash(value)] = embedding
            strings[hash(value)] = value
        else:
            print(f"Skipping non-string value: {value}")


    df_returns = df.copy()
    ## Convert the column to string indexes
    df_returns[column_name] = df_returns[column_name].apply(lambda x: string_ids.get(x, x))  # Use the string_ids mapping to convert to indexes

    return df_returns, embeddings, strings, ids_to_string

## Example usage
# anno_claims_df, embeddings, strings = transform_columns_to_embeddings(anno_claims_df, 'AccidentType')
# for key, value in embeddings.items():
#     print(f"Key: {key}, String: {strings[key]}")
#     print(f"Embedding shape: {value.shape}")
#     print(f"Embedding: {value}")
#     print("-----")

# %% [markdown]
# # Processing Data
# 

# %% [markdown]
# ## Pre-Pre-Processing
# This is where the pre-processing for the pre-processing happens. This includes refactoring the dataframe for analysis regarding pre-processing, for example dropping rows without SettlementValues and refactoring Exceptional_Circumstances to a binary True/False instead of a linguistic Yes/no.

# %%
def replace_yes_no_with_binary(df, column_names):
    """
    Replace 'Yes' and 'No' with 1 and 0 in the specified column of the DataFrame.
    """
    df_cpy = df.copy()
    df_cpy = df_cpy.dropna(subset=['SettlementValue'])


    ## replace Yes/No with True/False in the Yes_No columns
    for col in column_names:
        df_cpy[col] = df_cpy[col].replace({'Yes': 1.0, 'No': 0.0})
        # anno_claims_df[col] = anno_claims_df[col].astype('Int64')  # Convert to integer type

    return df_cpy

# %%
def replace_categorical_with_ints(df, column_names):
    """
    Replace categorical values with binary values in the specified column of the DataFrame.
    """
    df_cpy = df.copy()

    all_embeddings = {}
    all_embedded_strings = {}
    all_ids_to_string = {}
    ## convert the categorical columns to their hash values and embeddings
    for col in column_names:
        df_cpy, embeddings, strings, id_to_string = transform_columns_to_embeddings(df_cpy, col)

        all_embeddings.update(embeddings)
        all_embedded_strings.update(strings)

        if col not in all_ids_to_string:
            all_ids_to_string[col] = id_to_string
        else:
            all_ids_to_string[col].update(id_to_string)

    
    return df_cpy, all_embeddings, all_embedded_strings, all_ids_to_string


# %%
def map_genders(df, column_name):
    df_cpy = df.copy()

    mapped_gender = {}
    unique_values = df_cpy[column_name].unique()
    ##remove any null values from the unique values
    unique_values = unique_values[~pd.isnull(unique_values)]
    for i, value in enumerate(unique_values):
        mapped_gender[i] = value
        df_cpy[column_name] = df_cpy[column_name].replace(value, i)
    return df_cpy, mapped_gender

# %%
def convert_to_seconds(df, column_name):
    df_cpy = df.copy()
    # Check if the column exists in the DataFrame
    if column_name not in df_cpy.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # duplicate the column to avoid overwriting the original data
    df_cpy[column_name + '_time'] = df_cpy[column_name]
    # convert the original column to the day of the year (x/365)
    df_cpy[column_name] = pd.to_datetime(df_cpy[column_name], errors='coerce').dt.dayofyear
    # convert the time to seconds since start of the day (00:00:00)
    df_cpy[column_name + '_time'] = pd.to_datetime(df_cpy[column_name + '_time'], errors='coerce')
    df_cpy[column_name + '_time'] = df_cpy[column_name + '_time'].apply(
        lambda x: (x.hour * 3600 + x.minute * 60 + x.second) if pd.notnull(x) else 0
    )

    # convert the column to a float type
    df_cpy[column_name + '_time'] = df_cpy[column_name + '_time'].astype(float)

    return df_cpy


# %%
def convert_to_seconds_diff(df, col1, col2):
    df_cpy = df.copy()
    # Check if the columns exist in the DataFrame
    if col1 not in df_cpy.columns or col2 not in df_cpy.columns:
        raise ValueError(f"Columns '{col1}' or '{col2}' do not exist in the DataFrame.")

    # convert the columns to datetime
    df_cpy[col1] = pd.to_datetime(df_cpy[col1], errors='coerce')
    df_cpy[col2] = pd.to_datetime(df_cpy[col2], errors='coerce')

    # calculate the difference between the two columns
    df_cpy['days_diff'] = (df_cpy[col2] - df_cpy[col1]).dt.days.fillna(0).astype(int)

    # seconds_diff should be time from the start of the day (00:00:00)
    df_cpy['seconds_diff'] = df_cpy[col2].apply(
        lambda x: (x.hour * 3600 + x.minute * 60 + x.second) if pd.notnull(x) else 0
    )

    return df_cpy['days_diff'], df_cpy['seconds_diff']


# %%
def fix_prognosis(df, column_name):
    df_cpy = df.copy()
    # Check if the column exists in the DataFrame
    if column_name not in df_cpy.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # convert extract the month from the column (D. 4 months) -> 4
    df_cpy[column_name] = df_cpy[column_name].str.extract(r'(\d+)')[0].astype(float)
    # convert the column to numeric values
    df_cpy[column_name] = pd.to_numeric(df_cpy[column_name], errors='coerce')


    return df_cpy

# %%
def pre_pre_processing(df, yes_no_columns, categorical_columns):

    df_cpy = df.copy()

    ## replace the Yes/No columns with 1/0
    df_cpy = replace_yes_no_with_binary(df_cpy, yes_no_columns)

    ## replace the categorical columns with their hash values and embeddings
    df_cpy, all_embeddings, all_embedded_strings, all_ids_to_string = replace_categorical_with_ints(df_cpy, categorical_columns)

    df_cpy, gender_mapping = map_genders(df_cpy, 'Gender')


    df_cpy['Claim_Delay'], df_cpy['Claim_Delay_Time'] = convert_to_seconds_diff(df_cpy, 'Accident Date', 'Claim Date')  
    # drop claim date as it is no longer relevant as more meaningful data has been created from it
    df_cpy = df_cpy.drop(columns=['Claim Date'])

    df_cpy = convert_to_seconds(df_cpy, 'Accident Date')

    # fix the prognosis column to be numeric values
    df_cpy = fix_prognosis(df_cpy, 'Injury_Prognosis')

    return df_cpy, all_embeddings, all_embedded_strings, all_ids_to_string, gender_mapping


# %% [markdown]
# ## Pre-Processing
# This is where the data is imputed and duplicated to create more examples for training against different low-seen examples.

# %%
## seaborn heatmap of the missing values in the dataframe
plt.figure(figsize=(12, 8))
sns.heatmap(processed_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Anno Claims Dataframe')
##xticks at 45 degree angle
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
def impute_column_with_mlp(df, target_col):


    df_cpy = df.copy()
    
    if target_col not in df_cpy.columns:
        return df

    clean = df_cpy.dropna()
    # drop the tareget column from the dataframe
    Y = clean[target_col]
    X = clean.drop(columns=[target_col])


    ## split the data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    # print unique values in the target column
    print("Unique values in target column: ", Y_train.unique())
    print("Unique values in target column: ", len(Y_train.unique()))


    # Create and train the MLP model

    model_params = None
    model_opt = None
    model_compile = None
    callbacks = []
    # Define the model architecture
    if target_col in Yes_No_Columns:
        model_params = tf.keras.layers.Dense(1, activation='sigmoid')

        model_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        # compile docs state the order as (self, optimizer, loss, loss_weights, metrics, ...)
        model_compile = (model_opt, 'binary_crossentropy',['accuracy']) 

        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True))


    elif target_col in Categorical_Columns:
        model_params = tf.keras.layers.Dense(len(df[target_col].unique()), activation='softmax')
        

        model_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model_compile = (model_opt, 'sparse_categorical_crossentropy',['accuracy'])
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True))
        
    else:
        model_params = tf.keras.layers.Dense(1, activation='linear')
        model_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model_compile = (model_opt, 'mean_squared_error', ['mean_absolute_error'])
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True))
        
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        model_params
        
    ])
    model.compile(optimizer=model_compile[0], loss=model_compile[1], metrics=model_compile[2])


    model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1, callbacks=callbacks) #, class_weight=class_weights

    ## print the mse and mae of the model using keras
    X_pred = model.predict(X_test)

    if target_col in Categorical_Columns:
        X_pred = np.argmax(X_pred, axis=1)


    # # Impute missing values in the target column using the trained model
    for index, row in df_cpy.iterrows():
        if not pd.isnull(row[target_col]):
            continue

        ## set the row's target column to predicted value
        pred = model.predict(row.drop(target_col).values.reshape(1, -1))
        df_cpy.at[index, target_col] = np.argmax(pred, axis=1)
        
    return df_cpy


# %%
def impute_data(df, dic):
    df_cpy = df.copy()

    ## clean the data by dropping any rows with null values 


    ### loop through the columns and drop rows with null values
    for col in dic['drop']:
        if col not in df_cpy.columns:
            continue
        ## check if the column exists in the DataFrame
        if df_cpy[col].isnull().sum() > 0: 
            df_cpy = df_cpy.dropna(subset=[col])


    ## loop # rough the not needed columns and im# te them manually, treating null values in a float as 0.0, n# l binary values as False, integer as 0
    for col in dic['not_needed']:
        # col in df_cpy.columns:
        # check if the colu#  exists in the DataFrame
        if df_cpy[col].isnull().sum() == 0:  # Only impute if there are null values
            continue
        if col not in df_cpy.columns:
            continue

        if df_cpy[col].dtype == 'float64':
            df_cpy[col] = df_cpy[col].fillna(0.0)
        elif df_cpy[col].dtype == 'int32':
            df_cpy[col] = df_cpy[col].fillna(0)
        elif df_cpy[col].dtype == 'bool':
            df_cpy[col] = df_cpy[col].fillna(False)
        else:
            print(f"Column '{col}' has an unsupported data type for imputation.")

    for col in dic['needed']:
        if col not in df_cpy.columns:
            continue
        # ## loop through the needed columns and impute them with the trained imputing MLP model
        # Check if the column exists in the DataFrame
        if df_cpy[col].isnull().sum() == 0:  # Only impute if there are null values
            continue

        df_cpy = impute_column_with_mlp(df_cpy, col)

    
    return df_cpy



# %%
def refactor_model(df, categorical_columns, all_embeddings, all_string_ids):
    for col in categorical_columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame. Skipping...")
            continue
        print(f"Processing column: {col}")
        # Check if the column exists in the DataFrame
        category_strings = all_string_ids[col]
        print(f"Category strings: {category_strings}")
        category_embeddings = {}
        for key, value in category_strings.items():
            category_embeddings[key] = all_embeddings[hash(value)]
        print(f"Category embeddings: {category_embeddings}")

        print(category_embeddings)
        print(category_strings)


        # iterate through the column, add index imbeddings to the dataframe
        for index, row in df.iterrows():
            # Check if the value exists in the category_strings mapping
            if row[col] in category_strings.keys():
                # Get the corresponding embedding
                embedding = category_embeddings.get(row[col])
                if embedding is not None:
                    # Add the embedding to the DataFrame as new columns
                    for i, val in enumerate(embedding):
                        df.at[index, f"{col}_embedding_{i}"] = val
                else:
                    print(f"Embedding not found for value: {row[col]}")
            else:
                print(f"Value '{row[col]}' not found in category strings.")
        # Drop the original column after processing
        df = df.drop(columns=[col])





    return df, category_embeddings




# %% [markdown]
# # Train model

# %%
def train_model(df, target_col):
    df_cpy = df.copy()
    # Check if the column exists in the DataFrame
    if target_col not in df_cpy.columns:
        raise ValueError(f"Column '{target_col}' does not exist in the DataFrame.")
    
    # drop the tareget column from the dataframe
    Y = df_cpy[target_col]
    X = df_cpy.drop(columns=[target_col])


    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_percentage_error', patience=100, restore_best_weights=True)
    model.fit(X, Y, epochs=1000, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping]) # 

    return model

# %% [markdown]
# # Predictions
# 

# %%
def test_settlement_value(model, df, target_col):
    df_cpy = df.copy()
    # Check if the column exists in the DataFrame
    if target_col not in df_cpy.columns:
        raise ValueError(f"Column '{target_col}' does not exist in the DataFrame.")

    X = df_cpy.drop(columns=[target_col])

    predictions = model.predict(X)
    return predictions

# %%
processed_df, all_embeddings, all_embedded_strings, all_ids_to_string, gender_mapping = pre_pre_processing(base_claims_df, Yes_No_Columns, Categorical_Columns)
imputed_df = impute_data(processed_df, impution_dic)
refactored_df, category_embeddings = refactor_model(imputed_df, Categorical_Columns, all_embeddings, all_ids_to_string)
refactored_df


from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(refactored_df, test_size=0.2, random_state=42)

model = train_model(train_df, 'SettlementValue')

# %%
predictions = test_settlement_value(model, test_df, 'SettlementValue')
print("Predictions: ", predictions)
print("Y: ", test_df['SettlementValue'])



## score the model using mean absolute error and mean squared error from keras
mse = tf.keras.metrics.MSE(predictions, test_df['SettlementValue'])
mae = tf.keras.metrics.MAE(predictions, test_df['SettlementValue'])
mape = tf.keras.metrics.MAPE(predictions, test_df['SettlementValue'])

print("Average MSE: ", np.mean(mse))
print("Average MAE: ", np.mean(mae))
print("Average MAPE: ", np.mean(mape))

# %%
def predict_settlement_value(model, data):

    # Check if the data is a DataFrame or a single row
    if isinstance(data, pd.DataFrame):
        # If it's a DataFrame, use the model to predict on the entire DataFrame
        predictions = model.predict(data)
    else:
        # If it's a single row, reshape it to match the input shape of the model
        predictions = model.predict(np.array(data).reshape(1, -1))

    return predictions


