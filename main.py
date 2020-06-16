# -*- coding: utf-8 -*-
# Para crear la app web
from flask import Flask, render_template, request
# Para extraer tweets
import tweepy
import re
# Leer archivos csv
import pandas as pd
#----------------
import functools
import operator
# Para transformar de emoji a texto
import emoji
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from unicodedata import normalize
from datetime import timezone
import math
import numpy as np
from textblob import TextBlob
from googletrans import Translator
import random
import folium
import os
from werkzeug.utils import secure_filename
import geocoder
import nltk
nltk.download('stopwords')
app = Flask(__name__)       
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# settings
app.secret_key = "mysecretkey"
app.config['TEMPLATES_AUTO_RELOAD'] = True
# Carpeta de subida
app.config['UPLOAD_FOLDER'] = './Diccionario'

def autenticacion():
  consumer_key = "YBdxN84YmriV2Gq7nL31JvJxc"
  consumer_secret = "GTIuQumj7w89zDCmyGBDwanch7qAfBKq1ZGSV41LR2A8o8Gdfu"
  access_token = "1247276858187677698-J2uJvOU2jW7De05a7QNcpsImPnHTKA"
  access_token_secret = "82W737qtwPAgI4l7aHGjwcfXxGtxe2FqJfzMDRDcokGHF"
  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  return auth

def normalizar(dato):
  dato_split_emoji = emoji.get_emoji_regexp().split(dato)
  dato_split_whitespace = [substr.split() for substr in dato_split_emoji]
  dato_split = functools.reduce(operator.concat, dato_split_whitespace)  
  dato=convertir_emojis_en_sentiminetos(dato_split)
  dato= " ".join(dato_split)
  dato=dato.lower()
  dato=re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
    normalize( "NFD", dato), 0, re.I)
  dato=re.sub('[^a-z]+', ' ',dato)
  dato=dato.split()
  return dato

def eliminarurl(dato):
  result = re.sub(r"http\S+", "", dato)
  return result

def utc_to_local(utc_dt):
  return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)

dicc_emoji = []
df=pd.read_csv('./Diccionario/diccionario_emoji.csv',sep=',',encoding="utf-8")    
dicc_emoji=df.values.tolist()
sentimientos = []

df=pd.read_csv('./Diccionario/diccionario_sentimientos.csv',sep=';',encoding="utf-8")    
sentimientos.append(pd.unique(df['positivo']).tolist())
sentimientos.append(pd.unique(df['negativo']).tolist())

def convertir_emojis_en_sentiminetos(dato):
  dato = dato
  for j in range(len(dato)):
    for i in dicc_emoji:
      if i[0] == dato[j]:
        dato[j] = i[2]
  return dato

# eliminacion de stopwords
def stopw(data):
  a_stop = set(stopwords.words('spanish'))
  d_s =[]
  for word in data:
    if(word not in a_stop):
      d_s.append(word)
  return d_s
# stemmer
def steem(dato):
  stemmer = SnowballStemmer('spanish')
  stemmed = [stemmer.stem(item) for item in dato]
  return stemmed


# Distancia de jaccard
def jaccard(dic, tweet):
  temp = []
  coeficiente_pos = 0
  coeficiente_neg = 0
  positivas = set(dic[0])
  negativas = set(dic[1])
  b = set(tweet)
  var_norma = len(negativas)-len(positivas)
  res1 = len(positivas & b)
  num_elemnt1 = len(positivas)+len(b)+var_norma
  coeficiente_pos = round(res1/(num_elemnt1-res1),4)
  res2 = len(negativas & b)
  num_elemnt2 = len(negativas)+len(b)
  coeficiente_neg = round(res2/(num_elemnt2-res2),4)
  if coeficiente_pos > coeficiente_neg:
    temp.append(['POSITIVO', coeficiente_pos,coeficiente_neg])
  elif coeficiente_pos < coeficiente_neg:
    temp.append(['NEGATIVO', coeficiente_pos, coeficiente_neg])
  else:
    temp.append(['NEUTRO', coeficiente_pos, coeficiente_neg])
  return temp
"""
codigo coseno
"""
dicc=[]
def diccionario(data):
  for i in range(len(data)):
    if data[i] not in dicc:
      dicc.append(data[i])

def bolsa_y_df(datos):
  idf=[]
  df=[]
  tf=[]
  for i in range(len(dicc)):
    fd = 0
    frec = 0
    temp = []
    for j in range(len(datos)):
      frec = datos[j].count(dicc[i])
      if frec > 0:
        frec = 1+(math.log10(frec))
        fd = fd+1
      temp.append(frec)
    idf.append(math.log10(len(datos)/fd))
    df.append(fd)
    tf.append(temp)
  return idf,tf

qwe=[]
def fun_count(datos):
  del qwe[:]
  data=datos
  pos=[]
  neg=[]
  for i in range(len(data)):
    positivas=0
    for j in range(len(sentimientos[0])):
      positivas=positivas+data[i][1].count(sentimientos[0][j])
    pos.append(positivas)
  for i in range(len(data)):
    negativas=0
    for j in range(len(sentimientos[1])):
      negativas=negativas+data[i][1].count(sentimientos[1][j])
    neg.append(negativas)
  qwe.extend([pos,neg])
  return qwe

def fun_tf_idf(datos):
  idf,tf=bolsa_y_df(datos)
  tf_idf=[]
  for i in range(len(idf)):
    temp = []
    for j in range(len(tf[0])):
      temp.append(idf[i]*tf[i][j])
    tf_idf.append((temp))
  return tf_idf

def modulo(vector):
  sum = 0
  for i in vector:
    sum = sum+pow(i,2)
  math.sqrt(sum)
  return math.sqrt(sum)

def prod_esc(lista):
  temp2 = []
  for j in lista:
    modul=modulo(j)
    temp1=[]
    for i in j:
      temp1.append(i/modul)
    temp2.append(temp1)
  return temp2

def simi_cose(tweeet):
  datos = []
  temp = []
  long_norm = []
  datos.append(sentimientos[0])
  datos.append(sentimientos[1])
  diccionario(sentimientos[0])
  diccionario(sentimientos[1])
  for i in range(len(tweeet)):
    datos.append(tweeet[i][1])
  tf_idf=fun_tf_idf(datos)
  tf_idf = [list(f) for f in (zip(*tf_idf))]
  long_norm.extend(prod_esc(tf_idf))
  for i in range(len(long_norm[2:])):
    pos = round(np.dot(long_norm[0], long_norm[i+2]), 4)
    neg = round(np.dot(long_norm[1], long_norm[i+2]), 4)
    if pos > neg:
      temp.append([i, 'POSITIVO', pos, neg])
    elif pos < neg:
      temp.append([i, 'NEGATIVO', pos, neg])
    elif pos == neg:
      temp.append([i, 'NEUTRO', pos, neg])
  return temp
"""
-----------------------------------------
"""
twet_original = []
datos_limpios=[]
def buscar_tweets(num_Twets, consulta):
  del twet_original[:]
  del datos_limpios[:]
  tweet_tratado = []
  sim_cose = []
  api = tweepy.API(autenticacion(),wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
  temp = []
  i = 0
  clave="AIzaSyA_-paf4ASyR6yGGca9g1odrf5oiGNIm0o"
  for tweet in tweepy.Cursor(api.search, q=consulta+"-filter:retweets", geocode="-1.684495,-78.873492,318.51km", lang="es", tweet_mode="extended").items():
    tweet_location = ", ".join(normalizar(tweet.user.location))
    location = geocoder.google(tweet_location,key=clave).latlng 
    if location != "NULL" and tweet_location != '' and location != None :
      tweet_text = eliminarurl(tweet.full_text)
      if tweet_text not in temp:
        temp.append(tweet_text)
        i = i+1
        twet_original.append([i,location, utc_to_local(tweet.created_at), tweet_text])
        limpieza=steem(stopw(normalizar(tweet_text)))
        diccionario(limpieza)
        tweet_tratado.append([i, limpieza])
        if i == num_Twets:
          break
  datos_limpios.extend(tweet_tratado)
  sim_cose.extend(simi_cose(tweet_tratado))
  for i in range(len(tweet_tratado)):
    twet_original[i].extend(jaccard(sentimientos,tweet_tratado[i][1]))
    twet_original[i].append(sim_cose[i])
  del dicc[:]
  return twet_original

def tweets(data):
  del twet_original[:]
  tweet_tratado = []
  sim_cose = [] 
  del datos_limpios[:]
  for i in data:
    tweet_location = [i[1],i[2]]
    tweet_text = eliminarurl(i[4])
    twet_original.append([i[0],tweet_location,i[3],i[4]])
    limpieza=steem(stopw(normalizar(eliminarurl((tweet_text)))))
    diccionario(limpieza)
    tweet_tratado.append([i[0], limpieza])
    
  datos_limpios.extend(tweet_tratado)
  sim_cose.extend(simi_cose(tweet_tratado))
  fun_count(tweet_tratado)
  for i in range(len(tweet_tratado)):
    twet_original[i].extend(jaccard(sentimientos,tweet_tratado[i][1]))
    twet_original[i].append(sim_cose[i])
  del dicc[:]
  return twet_original

@app.route('/')
def Index():
  return render_template('index.html')

@app.route('/add_query', methods=['POST'])
def add_queryquery():
  if request.method == 'POST':
    consultaTw = request.form['consulta']
    numeroTw = request.form['numero']
    datos = buscar_tweets(int(numeroTw), consultaTw)           
    return render_template('consulta.html', datos=datos)

@app.route("/upload", methods=['POST'])
def upload():
  if request.method == 'POST':
    f = request.files['uploadfile']
    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    df=pd.read_csv('./Diccionario/1000tweet_clasificados.csv',sep=';',encoding="UTF-16")
    temp=[]
    temp=df.values.tolist()
    datos =tweets(temp)
    return render_template('consulta.html', datos=datos)

twet_original1=[]
def tweets1(data):
  del twet_original1[:]
  tweet_tratado = []
  sim_cose = []
  for i in data:
    tweet_location = [i[1],i[2]]
    tweet_text = eliminarurl(i[4])
    twet_original1.append([i[0],tweet_location,i[3],i[4]])
    limpieza=steem(stopw(normalizar(eliminarurl((tweet_text)))))
    diccionario(limpieza)
    tweet_tratado.append([i[0], limpieza])
  sim_cose.extend(simi_cose(tweet_tratado))
  fun_count(tweet_tratado)
  for i in range(len(tweet_tratado)):
    twet_original1[i].extend(jaccard(sentimientos,tweet_tratado[i][1]))
    twet_original1[i].append(sim_cose[i])
  del dicc[:]
  return twet_original1

@app.route("/regre", methods=['POST'])
def regre():
  if request.method == 'POST':
    df=pd.read_csv('./1000tweet_clasificados.csv',sep=';',encoding="UTF-16")
    temp=[]
    temp=df.values.tolist()
    datos=tweets1(temp) 
    y1=[]
    for i in range(len(temp)):        
      y1.append(temp[i][5])
    X = np.array(qwe).T
    y = np.array(y1)
    model = linear_model.LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1,max_iter=100, multi_class='ovr', n_jobs=1,penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)
    model.fit(X,y)
    predictions = model.predict(X)
    model.score(X,y)
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    name='Logistic Regression'
    kfold = model_selection.KFold(n_splits=10, shuffle=True,random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,cv=kfold, scoring='accuracy')
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    predictions = model.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    asd=fun_count(datos_limpios)
    for i in range(len(asd[0])):
      X_new = pd.DataFrame({'positivas':[asd[0][i]],'negativas':[asd[1][i]]})
      if  model.predict(X_new)[0]==1:
        print(twet_original[i][3],'    =>positivo')
        twet_original[i].append(model.predict(X_new)[0])
      if  model.predict(X_new)[0]==0:
        print(twet_original[i][3],'    =>negativo')
        twet_original[i].append(model.predict(X_new)[0])
          #del datos_limpios[:]
    datos=twet_original
    return render_template('regresion.html',datos=datos)    
@app.route('/grafica', methods=['POST'])
def grafica():
  if request.method == 'POST':
    total_pos=0
    total_nega=0
    total_neutro=0
    temp=[]
    if len(twet_original)>0:
      for i in range(len(twet_original)):
        total_neutro=total_neutro+twet_original[i][4].count('NEUTRO')
        total_pos=total_pos+twet_original[i][4].count('POSITIVO')
        total_nega=total_nega+twet_original[i][4].count('NEGATIVO')
      total_pos=(total_pos*100)/len(twet_original)
      total_nega=(total_nega*100)/len(twet_original)
      total_neutro=(total_neutro*100)/len(twet_original)
      temp.append([total_pos,total_nega,total_neutro])
    total_pos=0
    total_nega=0
    total_neutro=0
    if len(twet_original)>0:
      for i in range(len(twet_original)):
        total_neutro=total_neutro+twet_original[i][5].count('NEUTRO')
        total_pos=total_pos+twet_original[i][5].count('POSITIVO')
        total_nega=total_nega+twet_original[i][5].count('NEGATIVO')
      total_pos=(total_pos*100)/len(twet_original)
      total_nega=(total_nega*100)/len(twet_original)
      total_neutro=(total_neutro*100)/len(twet_original)
      temp.append([total_pos,total_nega,total_neutro])
    return render_template('grafica.html', datos=temp)


@app.route('/Textblob', methods=['POST'])
def Textblob():
  if request.method == 'POST':
    mapvar=folium.Map()
    mapvar = folium.Map(location=[-1.684495, -78.873492],zoom_start=6.5, width=700, height=600)
    for i in twet_original:
      tweet_text=stopw(normalizar(eliminarurl(i[3])))
      ubicacion=(i[1])
      translator = Translator()
      dato= " ".join(tweet_text)
      trad = translator.translate(dato, dest='en')
      feelings = TextBlob(trad.text)
      feelings = feelings.sentiment.polarity
      if feelings > 0:
        folium.Circle(location=[float(ubicacion[0]-random.uniform(0.1,-0.01)) ,float(ubicacion[1]-random.uniform(0.1,-0.01))],popup=i[3],radius=1000,color='green',fill=True, fill_color='green').add_to(mapvar)
      elif feelings < 0:
        folium.Circle(location=[float(ubicacion[0]-random.uniform(0.1,-0.01)), float(ubicacion[1]-random.uniform(0.1,-0.01))],popup=i[3],radius=1000,color='red',fill=True, fill_color='red').add_to(mapvar)
    item_txt = """<br> &nbsp; {item} &nbsp; <i class="fa fa-circle fa-2x" style="color:{col}"></i>"""
    html_itms = item_txt.format(col="red",item="Negativo")
    html_itms2 = item_txt.format(col="green",item="Positivo")
    ab= """ 
            <link rel="stylesheet" href="https://bootswatch.com/4/materia/bootstrap.min.css">
            <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
            <div div class="collapse navbar-collapse" id="navbarColor01">
                <br>
                <br>
                <h1 style="text-align: center;">ANÁLISIS DE SENTIMIENTOS CON TEXTBLOB</h1>
            </div>
            <input type="button" class="btn btn-primary" onclick="history.back()" name="volver atrás" value="volver atrás">
            <div
                style= "
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    width: auto;">
                """
    legend_html = """
            <div style="
            position: fixed; 
            bottom: 50px; left: 50px; width: 150px; height: 100px; 
            border:2px solid grey; z-index:9999; 

            background-color:white;
            opacity: .50;

            font-size:14px;
            font-weight: bold;

            ">
            &nbsp; {title} 

            {itm_txt}
            {itm_txt2}

            </div> """.format( title = "Leyenda del mapa", itm_txt= html_itms, itm_txt2= html_itms2)
    cd="""
        </div>
        """
    mapvar.get_root().html.add_child(folium.Element(ab))
    mapvar.get_root().html.add_child(folium.Element(legend_html))
    mapvar.get_root().html.add_to(folium.Element(cd))
    mapvar.save('./templates/textblob.html')  
    del mapvar
    return render_template('textblob.html')
app.run(host='0.0.0.0',port = 8080) 