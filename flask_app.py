from flask import Flask, request, jsonify
from text_preprocess import preprocess_text
from feature_extraction import feature_extraction
from predict import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def model_predict():
    # Get the input text from the form field 'text'
    input_text = request.form.get('text')  # Retrieve the 'text' form data

    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    # Process the text (e.g., preprocessing, feature extraction)
    processed_text = preprocess_text(input_text)
    pca_result = feature_extraction(processed_text)
    
    # Make prediction using the processed features
    cluster_num, cluster_name = predict(pca_result)

    # convert into real int not numpy int
    cluster_num = int(cluster_num)

    # Return the prediction results as a JSON response
    return jsonify({
        'cluster_number': cluster_num,
        'cluster_name': cluster_name
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
