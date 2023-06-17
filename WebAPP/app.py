from flask import Flask , render_template, url_for, request
import pandas as pd , numpy as np 
import pickle 

####################Loading the Model #######################

file_name = "model.pkl"
RF_model = pickle.load(open(file_name, 'rb'))# Reading the model from the disk


# Create a flask app 
app = Flask(__name__)


#Defining the homepage using our app 
@app.route('/')
def home():
    return render_template("home.html")
'''
in Flash there is two method POST and GET , since we are getting the independent values 
we shall use the POST 
'''
@app.route('/predict', methods = ['POST'])
def predict(): 
    if request.method =='POST':
        me = request.form['message']
        message = [float(x) for x in me.split()]
        vect = np.array(message).reshape(1, -1)
        my_prediction = RF_model.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__': 
	app.run(debug=True)