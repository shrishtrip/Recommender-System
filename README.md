# Recommender-System
Implementing and comparing various techniques for building a  Recommender System. 

## <b>---Recommender System using various approaches.---

<b> Loosely speaking, there are 2 broad ways:</b>
<ol>
  <b><li>Nearest Neighbour Approach</li></b>
  Find the k (as required) nearest neighbours to a given point. Then the characteristics of that point can be approximated to some
  function of the characteristics of the neighbours. This is quite intitive, but not always necessarily correct.
<b><li>Factorization Approach</li></b>
  As the Netflix Prize competition has demonstrated, matrix factorization models are superior to classic nearest-neighbor techniques for     producing product recommendations, allowing the incorporation of additional information such as implicit feedback, temporal effects, and   confidence levels.
</ol>

<b>Packages required to run the code:</b>
* numpy
* pandas
* math
* time
* scipy
* csv

After downloading the above packages, download this folder and make it the working directory. Then run the corresponding files for the differnet approaches:
* IIcollaborative.py-- for Item-Item Collaborative filtering
* IIcollaborativeWithBaseLine.py --for Item-Item Collaborative filtering with baseline approach
* UUcollaborative.py.py --for User-User Collaborative filtering
* UUcollaborativeWithBaseLine.py.py --for User-User Collaborative filtering with baseline approach
* Latent Factor Mosel.py --for Latent Factor Model

The comparison for all these in mentioned in the design doc.
Dataset was taken from https://grouplens.org/datasets/movielens/
