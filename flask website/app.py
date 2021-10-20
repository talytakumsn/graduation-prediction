from flask import Flask, render_template, request, redirect, url_for, session, make_response
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
import io
from io import StringIO
import csv
import category_encoders as ce


# load semua pickle
std = pickle.load(open('scaler.pkl', 'rb'))
reduce_D = pickle.load(open('reduce.pkl', 'rb'))
pca_model = pickle.load(open('pca_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

app = Flask(__name__)
app.secret_key = 'sidik'

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def index2():
    return render_template('index.html')

@app.route('/prediksi')
def prediksi():
    return render_template('prediksi.html')

@app.route('/eksekusi', methods=['POST'])
def home():
    jur = request.form['jur']
    jk = request.form['jk']
    usia = request.form['usia']
    asal = request.form['asal']
    jm = request.form['jm']
    gol = request.form['gol']
    org = request.form['org']
    ips1 = request.form['ips1']
    ips2 = request.form['ips2']
    ips3 = request.form['ips3']
    ips4 = request.form['ips4']
    ips5 = request.form['ips5']
    sks1 = request.form['sks1']
    sks2 = request.form['sks2']
    sks3 = request.form['sks3']
    sks4 = request.form['sks4']
    sks5 = request.form['sks5']
    sks6 = request.form['sks6']


    df = np.array([[jur, jk, usia, asal, jm, gol, org, ips1, ips2, ips3, ips4, ips5, sks1, sks2, sks3, sks4, sks5, sks6]])
    col = ['Jurusan', 'Jenis Kelamin', 'Usia', 'Asal', 'Jalur Masuk', 'Golongan UKT', 'Organisasi', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'SKS 1', 'SKS 2', 'SKS 3', 'SKS 4', 'SKS 5', 'SKS 6']
    
    df = pd.DataFrame(df, columns=col)

    # print(df)

    # masukkan ke pipeline pertama
    df = std.transform(df)

    # print(df)

    # reduksi dimensi
    df = np.dot(df,reduce_D.T)

    # prediksi pake model yang ada    
    pred = pca_model.predict(df)
    # print(pred)
    return render_template('hasil.html', data=pred)

@app.route('/visualisasi')
def visualisasi():
    return render_template('visualisasi.html')

@app.route('/masuk', methods=['GET', 'POST'])
def masuk():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin123':
            error = 'Username atau password Anda salah! Ulangi kembali'
        else:
            session['logged_in'] = True
            return redirect(url_for('admin'))
    return render_template('masuk.html', error=error)

@app.route('/admin')
def admin():
    if session.get('logged_in'):
        return render_template('a_beranda.html')
    else:
        return redirect(url_for("masuk"))

@app.route('/masterdata')
def masterdata():
    if session.get('logged_in'):
        masterdata = pd.read_pickle('data.pkl')
        return render_template('a_masterdata.html', tables=[masterdata.to_html(classes='data', header="true")])
    else:
        return redirect(url_for("masuk"))

@app.route('/visualisasidata')
def visualisasidata():
    if session.get('logged_in'):
        return render_template('a_visualisasidata.html')
    else:
        return redirect(url_for("masuk"))

@app.route('/prediksikelulusan')
def prediksikelulusan():
    if session.get('logged_in'):
        return render_template('a_prediksi.html')
    else:
        return redirect(url_for("masuk"))

@app.route('/hasilprediksi', methods=["POST"])
def hasilprediksi():
    if session.get('logged_in'):
        f = request.files['datafile']
        if not f:
            return "No file"

        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)
        #print("file contents: ", file_contents)
        #print(type(file_contents))
        
        col_y = ['Jurusan', 'Jenis Kelamin', 'Usia', 'Asal', 'Jalur Masuk', 'Golongan UKT', 'Organisasi', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPS 5', 'SKS 1', 'SKS 2', 'SKS 3', 'SKS 4', 'SKS 5', 'SKS 6']

        col_x = next(csv_input)
        rest = list(col_x)

        if col_x != col_y:
            error = 'Header Kolom Tidak Sesuai! Unggah file kembali dengan header kolom yang sesuai.'
        else:
            stream.seek(0)
            result = transform(stream.read())

            new_df = pd.read_csv(StringIO(result))
            # print(new_df)

            predict_df = encoder.fit_transform(new_df)
            # print(predict_df)

            predict_df = std.transform(predict_df)
            # print(predict_df)

            predict_df = np.dot(predict_df,reduce_D.T)

            new_predict = pca_model.predict(predict_df)
            # print(new_predict)

            new_df['Prediksi'] = new_predict.tolist()
            new_df['Prediksi'].replace(0, 'Terlambat', inplace=True)
            new_df['Prediksi'].replace(1, 'Tepat', inplace=True)
            # print(new_df)

            return render_template('a_hasil.html', tables=[new_df.to_html(classes='data', header="true")])
        
        return render_template('a_prediksi.html', error=error)
    else:
        return redirect(url_for("masuk"))

@app.route('/keluar')
def keluar():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)