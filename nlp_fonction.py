import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import wordninja
import pickle
import numpy as np
import warnings

# Ignorer les avertissements de seaborn
warnings.filterwarnings("ignore")


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

with open('top_5_tags.pkl', 'rb') as fichier:
    top_5_tags = pickle.load(fichier)

with open('liste_mot_long.pkl', 'rb') as fichier:
    liste_mot_long = pickle.load(fichier)

with open('liste_mot_unique.pkl', 'rb') as fichier:
    liste_mot_unique = pickle.load(fichier)

with open('liste_mots_10.pkl', 'rb') as fichier:
    liste_mots_10 = pickle.load(fichier)

with open('liste_tags_3_carac.pkl', 'rb') as fichier:
    liste_tags_3_carac = pickle.load(fichier)

with open('stop_words.pkl', 'rb') as fichier:
    stop_words = pickle.load(fichier)

with open('classes_correspondance.pkl', 'rb') as fichier:
    classes_correspondance = pickle.load(fichier)

with open('sgd_classifier_tfidf.pkl', 'rb') as model_file:
    loaded_sgd_classifier_tfidf = pickle.load(model_file)

with open('body_tfidf_vector.pkl', 'rb') as model_file:
    body_tfidf_vector = pickle.load(model_file)

    
#Fonction d'évaluation
def calculer_scores_jaccard(liste_tags_reels, liste_tags_predits):
    # Créez une liste vide pour stocker les scores de Jaccard
    scores_jaccard = []
    scores_precision = []
    tags_reels = []
    tags_predits = []

    # Boucle for pour itérer sur chaque élément de votre jeu de données
    for i in range(len(liste_tags_reels)):
        # Récupérez les ensembles de l'élément i
        ensemble1 = set(liste_tags_reels.iloc[i])
        ensemble2 = set(liste_tags_predits[i])
    
        # Calcul du score de Jaccard pour l'élément i
        intersection = len(ensemble1.intersection(ensemble2))
        union = len(ensemble1) + len(ensemble2) - intersection
    
        # Vérifiez si l'union est égale à zéro pour éviter la division par zéro
        if union != 0:
            score_jaccard = float(intersection) / union*100
        else:
            score_jaccard = 0.0

        # Calcul du score de précision pour l'élément i
        intersection = len(ensemble1.intersection(ensemble2))
        precision = 0
        if len(ensemble2) > 0:
            precision = (intersection / len(ensemble2)) * 100
        else:
            precision = 0.0
    
        # Ajoutez le score de Jaccard, les tags réels et les tags prédits aux listes correspondantes
        scores_jaccard.append(score_jaccard)
        tags_reels.append(ensemble1)
        tags_predits.append(ensemble2)
        scores_precision.append(precision)

    # Créez un DataFrame à partir des listes de tags réels, tags prédits et scores de Jaccard
    resultat_use_train = pd.DataFrame({'Tags Réels': tags_reels, 'Tags Prédits': tags_predits, 'Score Jaccard': scores_jaccard, 'Score de précision': scores_precision})

    # Calculez la moyenne des scores de Jaccard
    moyenne_jaccard_train = sum(scores_jaccard) / len(scores_jaccard)

    # Calculez la moyenne des scores de Jaccard
    moyenne_precision = sum(scores_precision) / len(scores_precision)

    # Affichez la moyenne des scores de Jaccard
    print("Moyenne des scores de Jaccard : ", moyenne_jaccard_train)

    # Affichez la moyenne des scores de Jaccard
    print("Moyenne des scores de précision : ", moyenne_precision)

    return resultat_use_train
#-----------------------------------------------------------------------------------------

def preditc_proba_seuil(predict_proba_liste, nb_classes, seuil_proba, seuil_min, classes_correspondance): 
    
    """
    Renseigner la variable issu du predict_proba, le seuil de probabilité d'appartenance à une classe 
    ainsi que la variable issu du mlb.classes_
    
    """
    # Création dataset vide qui va comporter toutes nos probas avec en ligne toutes nos observations et colonne nos classes
    df_probabilite = pd.DataFrame()

    # Boucle pour remplir le dataframe df_probabilities
    for i in range(nb_classes):
        # Accéder au bon élémenet issu de predict_proba (récupérer la proba d'appartenance à la classe)
        probabilite_i = predict_proba_liste[i][:, 1]
        df_probabilite[f"classe_{i}"] = probabilite_i
    
    df_probabilite = df_probabilite.set_axis(classes_correspondance, axis=1)

    # Créer un masque TRUE/FALSE avec le seuil de proba choisi
    df_mask = df_probabilite.apply(lambda x: x >= seuil_proba)
    
    # Comptez le nombre de True pour chaque ligne afin de savoir combien de proba sont supérieurs au seuil choisi et donc combien de classe possède l'observation
    nombre_true = df_mask.sum(axis=1)

    # Enregistrez cela dans un nouveau dataframe
    df_nb_true = pd.DataFrame({"nb_true": nombre_true})

    # Ajoutez une colonne pour la classe la plus probable 
    df_nb_true["nom_classe"] = df_probabilite.idxmax(axis=1)
    df_nb_true["proba_classe_max"] = df_probabilite.max(axis=1)
    
    def get_top_class(row):
        if row["nb_true"] == 0:
            # Aucune classe n'a une probabilité supérieure au seuil, on prend la classe avec la probabilité maximale
            max_proba_class = df_probabilite.loc[row.name].idxmax()
            max_proba_value = df_probabilite.loc[row.name].max()
            if max_proba_value >= seuil_min:
                return [max_proba_class]
            else:
                return "Pas de tags"  # Aucune classe ne dépasse le seuil minimum
        else:
            # Retourner toutes les classes avec une probabilité supérieure à 0.5
            classes_sup_05 = df_probabilite.columns[df_probabilite.loc[row.name] >= 0.5].tolist()
            return classes_sup_05

    df_nb_true["classe_predite"] = df_nb_true.apply(get_top_class, axis=1)
    
    return df_nb_true

#-------------------------------------------------------------------------------------------------

def process_text_2(text, rejoin=False, liste_mot_pas_assez_present=None, liste_3_caractere=None,remove_verbs_adverbs=False):
    """Fonction pour nettoyer un corpus de texte"""

    # Supprimer balise HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # Supprimer les liens
    text = re.sub(r"http\S+", "", text)

    # Supprimer les nombres
    text = re.sub(r'\d+', '', text)

    # Supprimer les _ entre des mots et les séparer
    text = re.sub(r'_', '', text)

    # Supprimer les caractères spéciaux
    if any(char in text for char in liste_3_caractere):
    # Si un caractère spécial est présent dans la liste liste_2_caractere, ne supprimez pas les caractères spéciaux du texte
        pass
    else:
    # Supprimer les caractères spéciaux du texte
        text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Mettre en minuscule tous les caractères et supprimer les espaces
    text = text.lower().strip()

    # Divisez le texte en mots uniquement si un mot a plus de 10 caractères
    if len(text) > 10:
        # Vérifier si un mot dans liste_mot_long est présent dans le texte
        if any(word in text for word in liste_mot_long):
            pass  
        else:
            text = separate_words(text)

    # Tokeniser le texte
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    liste_tokens = tokenizer.tokenize(text)

    # Supprimer les stopwords
    liste_tokens_clean = [w for w in liste_tokens if w not in stop_words]

    # Supprimer les mots présents que une seule fois
    liste_tokens_clean = [w for w in liste_tokens_clean if w not in liste_mot_unique]

    # Supprimer liste de mots pas assez présent 
    liste_tokens_clean = [w for w in liste_tokens_clean if w not in liste_mot_pas_assez_present]

    # Supprimer les mots de moins de 2 caractères sauf s'ils sont dans liste_tags_3_carac
    liste_tokens_clean = [w for w in liste_tokens_clean if len(w) > 3 or w in liste_3_caractere]

    # Supprimer les mots qui contiennent au moins deux traits de soulignement
    liste_tokens_clean = [w for w in liste_tokens_clean if "__" not in w]

    if remove_verbs_adverbs:
        # Identifier les parties du discours de chaque mot
        tagged_tokens = pos_tag(liste_tokens_clean)

        # Supprimer les mots qui sont des verbes (VB) ou des adverbes (RB)
        liste_tokens_clean = [w for w, pos in tagged_tokens if pos not in ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']]

    # Lemmatisation des mots (revenir à la racine du mot)
    trans = WordNetLemmatizer()
    trans_text = [trans.lemmatize(w) for w in liste_tokens_clean]

    # Supprimer les stopwords
    trans_text = [w for w in trans_text if w not in stop_words]

    # Renvoyer une liste de tokens (True) ou bien une chaîne de caractères avec chaque token séparé par un espace (False)
    if rejoin:
        return " ".join(trans_text)

    return trans_text

#-------------------------------------------------------------------------------------------------------

def clean_data_sup_verbe(doc):
    new_doc = process_text_2(doc
                             ,rejoin=True
                             ,liste_mot_pas_assez_present = liste_mots_10
                             ,liste_3_caractere = liste_tags_3_carac 
                             ,remove_verbs_adverbs=True)
    return new_doc


def separate_words(text):
    # Utiliser wordninja pour séparer les mots collés
    words = wordninja.split(text)
    return ' '.join(words)

#-------------------------------------------------------------------------------------------
def predict_tags_tfidf(titre, body):


    df_prep = pd.DataFrame(columns=['titre_clean', 'body_clean'])

    df_prep['titre_clean'] = [titre]
    df_prep['body_clean'] = [body]

    df_prep['titre_clean'] = df_prep['titre_clean'].apply(clean_data_sup_verbe)
    df_prep['body_clean'] = df_prep['body_clean'].apply(clean_data_sup_verbe)
    df_prep['title_bode_clean'] = df_prep['titre_clean']+df_prep['body_clean']

    body_tfidf_test = body_tfidf_vector.transform(df_prep['title_bode_clean'])


    # Obtenez les probabilités prédites pour chaque classe pour l'ensemble d'entraînement
    probabilities_test_bert = loaded_sgd_classifier_tfidf.predict_proba(body_tfidf_test)

    predict_proba_liste = probabilities_test_bert
    nb_classes = 200
    seuil_proba = 0.5
    seuil_min = 0.1
    #classes_correspondance = mlb.classes_

    prediction_bert_svm_test = preditc_proba_seuil(predict_proba_liste, nb_classes, seuil_proba, seuil_min, classes_correspondance)

    prediction_tags = prediction_bert_svm_test['classe_predite']

    return prediction_tags[0]