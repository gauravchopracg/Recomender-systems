
from flask import Flask, request, jsonify, render_template
import pickle
import scipy.sparse as sparse
from Recommendation import recommendation

app = Flask(__name__)
model_var = pickle.load(open('model.pkl', 'rb'))




@app.route('/')
def predict():
    
    
    person_vecs = sparse.csr_matrix(model_var.user_factors)
    content_vecs = sparse.csr_matrix(model_var.item_factors)
    person_id = 50
    mat=recommendation.sparse_person_content
    recommendations = recommendation.recommend(person_id, mat, person_vecs, content_vecs)
    return render_template('index.html', prediction_text= recommendations['title'])



if __name__ == "__main__":
    app.run(debug=True)
