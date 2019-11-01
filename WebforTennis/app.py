from flask import Flask, render_template, redirect
from flask_pymongo import PyMongo
from flask import Flask, request, make_response
import predict_video
import pymongo

# Create an instance of Flask
app = Flask(__name__)

# Use PyMongo to establish Mongo connection
mongo = PyMongo(app, uri="mongodb://localhost:27017/mission_to_mars_app")


# Create connection variable
#conn = 'mongodb://localhost:27017'

# Pass connection to the pymongo instance.
#client = pymongo.MongoClient(conn)

# Connect to a database. Will create one if not already available.
#db = client.mars_app


# Route to render index.html template using data from Mongo
@app.route("/")
def home():

    # Find one record of data from the mongo database
    destination_data = mongo.db.mars_data.find_one()

    # Return template and data
    return render_template("index.html", vacation=destination_data)


# Route that will trigger the scrape function
@app.route("/", methods = ['POST'])
def predict():

    # Run the scrape function and save the results to a variable
    # @TODO: YOUR CODE HERE!

    fileName = request.form.get('video')
    modelName = request.form.get('model1')

    print('filename = ', fileName)
    print('modelName = ', modelName)

    results = predict_video.predict(modelName,fileName)
    #results = {'height':20}

    print(results)

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
