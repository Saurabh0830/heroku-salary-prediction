import re
from flask import Flask ,render_template,request
import joblib

app = Flask(__name__)

l=joblib.load("predict.pkl")

@app.route('/')
def welcome():
    return render_template("base.html")

@app.route('/predict' , methods=['post'])
def predict():
    a=request.form.get("experience")
    b=request.form.get("test_score")
    c=request.form.get("interview_score")
    
    salary=l.predict([[int(a),int(b),int(c)]])

    return render_template("base.html", prediction="predicted salary is : {}".format(round(salary[0],2)))

if __name__=='__main__':
    app.run(debug=True)