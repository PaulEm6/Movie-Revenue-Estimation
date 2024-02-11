from flask import Flask, request, jsonify
from flask_cors import CORS
from api_module import predict


app = Flask(__name__)
CORS(app)

# ... (your existing code)

@app.route('/', methods=['POST'])
async def receive_data():
    data = request.json
    # Assuming the existing code for actor casting and other processing

    # Extract data needed for the predict method
    year = int(data['year'])
    '''
    budget = int(data['budget'])
    duration = int(data['duration'])
    genres = data['Genres']
    mpaa_rating = data['MPAA_rating']
    keywords = data['Keywords']
    source = data['Source']
    production_method = data['Production_Method']
    creative_type = data['Creative_type']
    countries = data['Countries']
    '''
    
    # Call the predict method
    prediction_result = predict(year)
    #prediction_result = predict(year, budget, duration, genres, mpaa_rating, keywords, source, production_method, creative_type, countries)

    # Construct the JSON response
    result = {
              "prediction_result": prediction_result}

    # Return the result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=8080)
