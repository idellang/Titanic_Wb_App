import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from PIL import Image
from preprocessing_classes import DataFrameSelector, RelAdder, MostFrequentImputer

# load the test data. This is needed for preprocessing
test_data = pd.read_csv('data/test.csv')

# load sklearn classes


image = Image.open('images/titanic_cutout.png')
st.image(image, width=300)

st.title("Would you have survived the Titanic tragedy?")
st.markdown('''
    This app predicts whether you will survive the Titanic crash based on your input features
''')

# about page
expander_bar = st.expander('About')
expander_bar.markdown("""
* **Python libraries:** pandas, streamlit, numpy, matplotlib, PIL
* **Data source:** [Titanic Dataset](https://www.kaggle.com/c/titanic/data).
* **Github repository:** [Titanic_Wb_App](https://github.com/idellang/Titanic_Wb_App)
""")


st.header('Input your details')


def user_input_features():
    Name = st.text_input('Name: ', 'Jonathan Berg')
    Pclass = st.selectbox(
        'Ticket Class: ', ['1st class', '2nd class', '3rd class'])
    Sex = st.selectbox('Sex: ', ['male', 'female'])
    Age = st.number_input('Age', int(test_data['Age'].min(
    )), int(test_data['Age'].max()), int(test_data['Age'].median()))

    SibSp = st.number_input('Number of siblings/spouse on board', test_data['SibSp'].min(
    ), test_data['SibSp'].max(), int(test_data['SibSp'].median()))

    Parch = st.number_input('Number of parents/children on board', test_data['Parch'].min(
    ), test_data['Parch'].max(), int(test_data['Parch'].median()))

    Fare = st.number_input('Fare: ', test_data['Fare'].min(
    ), test_data['Fare'].max(), test_data['Fare'].median())
    Embarked = st.selectbox('Port of Embarkation: ', [
                            'Cherbourg', 'Queenstown', 'Southampton'])

    input_data = {
        'Pclass': int(Pclass[0]),
        'Name': Name,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked[0]
    }

    input_features = pd.DataFrame(input_data, index=[0])
    return input_features


input_features = user_input_features()
st.write('---')
st.header('User Input Parameters')
st.write('''Please check if the following details are correct: \n''')
st.write(f'''
    * Name: {input_features['Name'].values[0]}\n
    * Ticket Class: {input_features['Pclass'].values[0]}\n
    * Sex: {input_features['Sex'].values[0]}\n
    * Age: {input_features['Age'].values[0]}\n
    * Siblings/spouses on board: {input_features['SibSp'].values[0]}\n
    * Parents/children on board: {input_features['Parch'].values[0]}\n
    * Fare: {input_features['Fare'].values[0]}\n
    * Port of Embarkation: {input_features['Embarked'].values[0]}\n

''')
st.write('---')

submit_button = st.button('Make Predictions')


def make_predictions():
    # combine the input feature to test
    combined = pd.concat([input_features, test_data])

    load_model = load(open('rf_gridsearch.pkl', 'rb'))
    load_preprocess_pipeline = load(open('preprocess_pipeline.pkl', 'rb'))

    test_prepoc = load_preprocess_pipeline.fit_transform(combined)[0]
    prediction = load_model.predict([test_prepoc])
    proba = load_model.predict_proba([test_prepoc])

    return prediction, proba


if submit_button:
    prediction, proba = make_predictions()

    if prediction[0] == 0:
        image = Image.open('images/drown.jpg')
        image = image.resize((250, 250))
        st.image(image)
        st.write(
            f'You have {round((proba[0][1]) * 100, 2)}% chance of survival')
    elif prediction[0] == 1:
        image = Image.open('images/survived.jfif')
        image = image.resize((250, 250))
        st.image(image)
        st.write(
            f'You have {round((proba[0][1]) * 100, 2)}% chance of survival')
else:
    st.write('Click to make predictions')
