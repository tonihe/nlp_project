import streamlit as st 
import numpy as np
import pandas as pd
import nltk 
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
import re
import datasets 
import matplotlib.pyplot as plt
import sklearn 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import pickle

####################################
            # Get Dataset #
####################################
@st.cache 
def get_data():
    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')   
    df = dataset['train'].to_pandas()

    return df

df = get_data()

####################################
        # Basic Section Layout #
####################################

header = st.container()
dataset = st.container()
preprocess = st.container()
features = st.container()
model_training = st.container()

####################################
            # Header Section #
####################################

with header: 
    st.title("Hate Speech Detector")

####################################
    # Model Select Sidebar #
####################################

    st.sidebar.write("""# Select ML Model """)
    select_model = st.sidebar.selectbox("", ("","Logistic Regression", "Naive Bayes", "Linear SVC"))

####################################
    # The Dataset Used #
####################################
with dataset:
    if select_model == "":
        st.header("Dataset")
        st.write("\"This is a public release of the dataset described in Kennedy et al. (2020), consisting of 39,565 comments annotated by 7,912 annotators, for 135,556 combined rows. The primary outcome variable is the 'hate speech score' but the 10 constituent labels (sentiment, (dis)respect, insult, humiliation, inferior status, violence, dehumanization, genocide, attack/defense, hate speech benchmark) can also be treated as outcomes. Includes 8 target identity groups (race/ethnicity, religion, national origin/citizenship, gender, sexual orientation, age, disability, political ideology) and 42 identity subgroups.\"")
        st.write("Original Source [link](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech#key-dataset-columns)")
        
        st.dataframe(df)

    elif select_model == "Logistic Regression":
        st.header("Logistic Regression")
        st.write("A generalized linear algorithm used to predict categorical probability of a class or event. Logistic regression is often used in categorical classification problems. In our case, the goal is simple, we've split our samples into hatespeech or not hatespeech. Logistic regression model will help us calculate the probability of a piece of text belonging in a category. We can use logistical regression as a rudimentary way to figure out whether or not a piece of text is hatespeech.")

    elif select_model == "Naive Bayes":
        st.header("Naive Bayes")
        st.write("Naive Bayes algorithm applies Bayes theorem to solve problems and 'naively' assumes that features are independent of eachother. Naive Bayes is commonly used in NLP as it can classify text with speed and decent accuracy. In our case, Multinomial Naïve Bayes considers the frequency in which a feature vector appears and makes assumptions accordingly. The algorithm can process large amounts of data in shorter amount of time compared to other methods. ")

    elif select_model == "Linear SVC":
        st.header("Linear SVC")
        st.write("Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.  LinearSVC is a fast implementation of Support Vector Classification for the case of a linear kernel.")
        st.write("Source sklearn [link](https://scikit-learn.org/stable/modules/svm.html)")
with preprocess:

    st.header("Data Exploration")
    st.write("Using the dataset provided by ucberkeley-dlab, it is found that out of the entire dataset of 135556 rows, 59.5% of the data is not classified as hatespeech, while 40.5% of the data is considered hatespeech by the annotators.")
    df_focus = df[['hatespeech', 'text']]
    df_focus.loc[df_focus['hatespeech'] == 1.0] = 2.0

    labels = ['none-hatespeech 0.0', 'hatespeech 1.0']
    explode = [0.075, 0]
    sizes = [len(df_focus[df_focus['hatespeech'] == 0.0]), len(df_focus[df_focus['hatespeech'] == 2.0])]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

    st.write("Shape of the dataframe", df_focus.shape)
    st.write("Split between none-hate: 0 and hatespeech: 1", sizes)

with features:
    st.header("Features")

    features = df_focus.iloc[:,1].values
    labels = df_focus.iloc[:,0].values

    #load the stopwords
    stopword = stopwords.words('english')

    #load lemmatizer
    from nltk.stem import WordNetLemmatizer
    lemma = WordNetLemmatizer()

    processed_features = []

    for sentence in range(0, len(features)):
        # remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
        
        # remove all single characters
        processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        processed_feature = lemma.lemmatize(processed_feature)

        processed_feature = processed_feature.lower()

        processed_features.append(processed_feature)

    st.write(""" #### examples of features after text pre-processing: """)
    st.write(processed_features[0:3])

### TDIF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.8)
pickle.dump(vectorizer, open("vectorizer.pickle", "wb")) #Save vectorizer
pickle.load(open("vectorizer.pickle", 'rb'))      #Load vectorizer
processed_features = vectorizer.fit_transform(processed_features).toarray()

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.3, random_state=0)

with model_training:
####################################
    # Hatespeech Function #
####################################
    def is_it_hatespeech(x):
            if x == 0.0: 
                st.sidebar.success("Congratulations, this text does not contain hatespeech!")
            else:
                st.sidebar.error("This text contains hatespeech! You might want to doublecheck this text before you post!")

####################################
    # Selection of Algorithms #
####################################
def select_classifier_ui(clf_name, X_train, X_test, y_train, y_test):
    if clf_name == "Logistic Regression":
        st.sidebar.write("Model selected: ", select_model)
        model = LogisticRegression().fit(X_train, y_train)
        predicted_classes = model.predict(X_test)
        logistic_report = classification_report(y_test, predicted_classes)
        st.sidebar.text(logistic_report)
        return model

    elif clf_name == "Naive Bayes":
        st.sidebar.write("Model selected: ", select_model)
        model = MultinomialNB().fit(X_train, y_train)
        predicted_classes = model.predict(X_test)
        nb_report = classification_report(y_test, predicted_classes)
        st.sidebar.text(nb_report)
        return model

    elif clf_name == "Linear SVC":
        st.sidebar.write("Model selected: ", select_model)
        model = LinearSVC().fit(X_train, y_train)
        predicted_classes = model.predict(X_test)
        svm_report = classification_report(y_test, predicted_classes)
        st.sidebar.text(svm_report)
        return model

model = select_classifier_ui(select_model, X_train, X_test, y_train, y_test)
user_input = ["This is just an example of user input"]
dummy_text = "This is dummy text."

if select_model != "":
        text_input = st.sidebar.text_area("Check for Hatespeech", dummy_text, height=200)
        def classify_text(text_input):
            user_input[0] = text_input
            pickle.load(open("vectorizer.pickle", 'rb'))      #Load vectorizer
            x = vectorizer.transform(user_input)
            st.sidebar.write("User Input: " , user_input)
            st.sidebar.write(model.predict(x))
            st.sidebar.write(is_it_hatespeech(model.predict(x)))

        st.sidebar.button("Analyze Text", on_click=classify_text(text_input))