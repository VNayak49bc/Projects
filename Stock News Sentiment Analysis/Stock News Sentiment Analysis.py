from flask import Flask,render_template,request
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd

#load the model from disk

def sentiment(text):
    global analyser
    try:
        sentence = Sentence(text)
        analyser.predict(sentence)
    except:
        analyser = TextClassifier.load('en-sentiment')
        sentence = Sentence(text)
        analyser.predict(sentence)
    finally:
        return str(sentence.labels[0])[:-9]

def news(companies):
        if companies.split() != 1:
            companies = [i.upper() for i in companies.strip().split()]
        else:
            companies = [companies.upper()]
        finviz_url = 'https://finviz.com/quote.ashx?t='
        news_tables = {}
        for company in companies:
            url = finviz_url + company
            req = Request(url=url, headers={'user-agent': 'my-app'})
            response = urlopen(req)
            html = BeautifulSoup(response, features='html.parser')
            news_table = html.find(id='news-table')
            news_tables[company] = news_table

        parsed_data = []
        for company, news_table in news_tables.items():
            for row in news_table.findAll('tr'):
                title = row.a.text
                date_data = row.td.text.split(' ')
                if len(date_data) != 1:
                    date = date_data[0]
                parsed_data.append([company, date, title])

        df = pd.DataFrame(parsed_data, columns=['Company', 'Date', 'Headline'])
        df['Date'] = pd.to_datetime(df.Date).dt.date
        df['Sentiment'] = df['Headline'].map(sentiment)
        result = pd.DataFrame(columns=df.columns)
        for i in df['Company'].unique():
            result = pd.concat([result, df[df['Company'] == i].iloc[:10,:]])           
        result = result.reset_index(drop=True)
        return result

app_1 = Flask(__name__)

@app_1.route('/')
def home():
    return render_template('home.html')

@app_1.route('/predict',methods=['POST'])
def predict():
    
    if request.method=='POST':
        companies = request.form['review']

        result = news(companies)

        return render_template('result.html',tables=[result.to_html(classes='data', header="False")])
    
if __name__=='__main__':
    app_1.run(debug=True)