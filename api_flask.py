from nlp_fonction import *
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/', methods=['POST'])
def prediction_tags():
    #Récupérer les inputs du l'utilisateur
    titre_test = request.json.get('titre_test')
    body_test = request.json.get('body_test')
    
    if titre_test is None or body_test is None:
        return jsonify({'error': 'Les champs titre_test et body_test sont requis'}), 400

    resultat = predict_tags_tfidf(titre_test, body_test)

    response_message = f"Les tags prédits sont : {resultat}"

    return jsonify({'message': response_message})

if __name__ == '__main__':
    app.run(host= "0.0.0.0", port=8080, debug=True)