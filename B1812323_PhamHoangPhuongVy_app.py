import B1812323_PhamHoangPhuongVy_train 
from flask import Flask, request, render_template 

app = Flask(__name__,template_folder="web")

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/classify',methods=['GET'])
def classify():
    try:
        sepal_len = request.args.get('slen') 
        sepal_wid = request.args.get('swid')
        petal_len = request.args.get('plen') 
        petal_wid = request.args.get('pwid') 

        variety = B1812323_PhamHoangPhuongVy_train.classify(sepal_len, sepal_wid, petal_len, petal_wid)

        return render_template('output.html', variety=variety)
    except:
        return render_template('Error.html')

if(__name__=='__main__'):
    app.run(debug=True)