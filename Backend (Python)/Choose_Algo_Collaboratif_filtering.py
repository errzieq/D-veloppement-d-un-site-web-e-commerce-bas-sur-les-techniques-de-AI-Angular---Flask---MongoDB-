# Nettoyage des donn�es:
import pandas as pd 
import numpy as np
import random
random.seed(9001) #pour ne pas avoir toujours les memes erreurs � chaque fois qu'on re ex�cute le projet.
import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client[ "PFE" ]

#rating
ratings = db[ "metauser" ]
tab = pd.DataFrame(list(ratings.find()))
tab = tab.drop('_id', 1)
tab = tab.drop('like', 1)
tab = tab[tab.rating != "-"]

tab.columns

tab['userId'] = tab['userId'].astype(int)
useri,frequsers=np.unique(tab.userId,return_counts=True)#useri les id des users, frequsers les freq de chaque user
itemi,freqitems=np.unique(tab.produitId,return_counts=True)#itemi les id des item, freqitem les freq de chaque item
n_users=len(useri)
n_items=len(itemi)
print("le nombre des utilisateurs est :"+ str(n_users) + " Et le nombre des items est: "+ str(n_items))


#Un des probl�mes qu'on a rencontr� �tait le fait que les ids des produits n'�tait pas ordonn�e.
#C'est � dire on peut trouver l'utilisateur 1,2,3,5 et 8 sans trouver les utilisateurs 4 ,6 et 7.
#Ceci nous a pos� un probl�me dans la cr�ation de la matrice user-produit parce que on risque d'avoir plusieurs lignes et colonnes vides.
#Pour ca, on a cr�e un tableau indice_user et un tableau indice_produit qui contiennent les anciens id et les nouvelles id par expl (1,2,5,6)=>(1,2,3,4) 
#puis on a ajout� deux colonnes sur le tableau principale qui contient ces nouveaux IDs.
#on a export� les nouvelles ids dans un fichier csv, et � chaque fois on utilise ce nouveau fichier.

indice_user = pd.DataFrame()
indice_user["indice"]=range(1,len(useri)+1)
indice_user["useri"]=useri

indice_item = pd.DataFrame()
indice_item["indice"]=range(1,len(itemi)+1)
indice_item["itemi"]=itemi

#cr�er user_ID_new et Item_ID_new
x=[]
y=[]

for i in range(0,len(tab)):
    x.append((indice_user.indice[indice_user.useri==tab.userId.iloc[i]].axes[0]+1)[0])
    y.append((indice_item.indice[indice_item.itemi==tab.produitId.iloc[i]].axes[0]+1)[0])

tab["userIdnew"]=x
tab["produitIdnew"]=y

tab[:20]

#validation croisé:
from sklearn import model_selection as cv
train_data, test_data = cv.train_test_split(tab[["userIdnew","produitIdnew","rating"]], test_size=0.25,random_state=123)

 
# La r�gle dit que si on a une grande sparsity (ou bien raret� des donn�es, c'est de ne pas arriver � calculer la 
# similarit� entre 2 utilisateurs par expl si chaqu'un a aim� different items que l'autre), les mod�les Model Based seront les plus efficace.
# Calculant alors la sparsity:

sparsity=round(1.0-len(tab)/float(n_users*n_items),3)
print('The sparsity level of our data base is ' +  str(sparsity*100) + '%')

# La pourcentage de sparsity est bien grande donc, on peut confirmer d�s maintenant que les mod�les Model Based seront les 
# mod�les les plus efficaces.

# # 1. Memory based Collaboratif Filtering:
# ### 1.1 La mise en place du mod�le:
# On va commencer par cr�er les mod�les Memory based.
#     -Les mod�les User-Based: "Les utilisateurs qui sont similaires � vous, ont aim� aussi ..."  
#     -Les mod�les Item-Based: "Les utilisateurs qui ont aim� ca, ont aim� aussi ..."  
# Pour expliquer plus:
# -le modele User-Based: va prendre un utilisateur, trouve les utilisateur les plus similaires � lui en se basant sur la note, Puis recommande les items aim� par ces utilisateurs(ca prend un user et retourne des items)
# -Le modele Item-Based: prend un item, cherche les utilisateurs qui ont aim� cet item, trouve les items aim� par ces utilisateurs
# (ca prend un item et retounes une liste des items)
# Pour le faire, on utilis� 2 m�triques le cosine similaire et cityblock.
# Pour le faire, on a commenc� par cr�er les matrice user-item train et test. Ce sont les deux matrices qui vont crois� les notes de utilsiateurs et des items.
# Puis, on a cr�e nos 4 mod�les Memory Based  
# � la fin, on a cr�e une fonction pour faire les pr�dictions selon le mod�le

train_data_matrix = np.zeros((n_users, n_items))#matrice nulle de longuer tous les users et tous les items
for line in train_data.itertuples():#parcourire la ligne col par col
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#calcule de la cos similarity : (construction du mod�le)
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
user_similarity1 = pairwise_distances(train_data_matrix, metric='cityblock')
item_similarity1 = pairwise_distances(train_data_matrix.T, metric='cityblock')

def predict(ratings, similarity, type='user'):#prend
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)#mean pour chauqe utilisateur (type = float)
        #np.newaxis pour convertir mean_user_rating de array de float en array d'array pour l'utiliser avec ratings
        #puis on a normalis� la var ratings (rating - E)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) #(type === array comme la var rating)
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        
    x = np.zeros((n_users, n_items))
    for i in range(0,n_items):
        a=max(pred[:,i])
        b=min(pred[:,i])
        c=0 # min rating
        d=5 # max rating
        for j in range(0,n_users):
            x[j,i]=(pred[:,i][j]-(a-c))*d/(b-a+c)
    
    return x

#la pr�diction avec les differents mod�les:
item_prediction = predict(test_data_matrix, item_similarity, type='item')
user_prediction = predict(test_data_matrix, user_similarity, type='user')
item_prediction1 = predict(test_data_matrix, item_similarity1, type='item')
user_prediction1 = predict(test_data_matrix, user_similarity1, type='user')

#1.2. La comparaison des RMSE:
#la creation de la fonction qui calcule le RMSE:
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth): #Root Mean Squared Error
    prediction = prediction[ground_truth.nonzero()].flatten() 
    #.flatten() fusionne les elts des array en un array
    #on attribue a prediction, les r�sultats des pr�dictions o� on connait le vrais rating cad:
    #prediction: tous nos pr�dictions sur test; ground_truth.nonzero():les vrais r�sultats qu'on a dans test
    #on va mettre dans prediction les valeurs qu'on a pr�dit pour les elts qu'on a d�ja.
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()#pareil dans ground truth
    return sqrt(mean_squared_error(prediction, ground_truth))

print ('User-based CF: The RMSE for the cosine similarity metric is : ' + str(rmse(user_prediction, test_data_matrix)))
print ('Item-based CF: The RMSE for the cosine similarity metric is :  ' + str(rmse(item_prediction, test_data_matrix)))
print ('User-based CF: The RMSE for the cityblock similarity metric is :   ' + str(rmse(user_prediction1, test_data_matrix)))
print ('Item-based CF: The RMSE for the cityblock similarity metric is : ' + str(rmse(item_prediction1, test_data_matrix)))

# Le meilleur mod�le est celui qui a le RMSE le plus petit.
# Pour notre cas c'�tait User based pour la m�trique cosine.

# Conclusion:     
# -Les mod�les Memory based sont  facile � implementer et g�n�re des bonnes r�sultats.
# -Ce type de mod�le n'est pas scalable (n'est pas vraiment pratique dans un probl�me d'une grande base de donn�e vu qu'il 
# (lorsqu'on commence avec un nouveau utilisateur/item dont on n'a pas assez d'information) calcule � chaque fois la corr�lation entre tous les utilisateur / les items) et ne resolut pas le probl�me de cold start
# Pour r�pondre au probl�me de scalability on cr�e les modele Model Based(partie suivante).
# Pour r�pondre au probl�me de cold start, on utilise la recommandation Content based (on va pas l'utiliser vu qu'on n'a pas ces donn�es )

# # 2. Model-based Collaborative Filtering

# Dans cette partie du projet, nous appliquons le deuxi�me sous-type du fitrage collaboratif : "Model-based". 
# Il consiste � appliquer la matrice de factorisation (MF) : c'est une m�thode d'apprentissage non supervis� de d�composition
# et de r�duction de dimensionnalit� pour les variables cach�es. 
# Le but de la matrice de factorisation est d'apprendre les pr�f�rences cach�es des utilisateurs et les attributs cach�s des items
# depuis les ratings connus dans notre jeu de donn�es, pour enfin pr�dire les ratings inconnus en multipliant les matrices de varibales cach�es des utilisateurs et des items. 
# Il existe plusieurs techniques de r�duction de dimensionnalit� dans l'impl�mentation des syst�mes de recommendations. 

# Dans notre projet, nous avons utilis� :
# - SVD : singular value decomposition
# - SGD : Stochastic Gradient Descent
# - ALS : Alternating Least Squares

#2.1 Singular value decomposition (SVD)
#2.1.1 La mise en place des SVD:

# Cette technique, comme toutes les autres, consiste � r�duire la dimensionnalit� de la matrice User-Item calcul�e pr�cedemment.
# Posons R la matrice User-Item de taille m x n (m : nombre de users, n: nombre d'items) et  k: la dimension de l'espace des caract�res cach�s.
# L'�quation g�n�rale de SVD est donn�es par : R=USV^T avec:
# - La matrice U des caract�res cach�s pour les utilisateurs : de taille m*k
# - La matrice V des caract�res cach�s pour les items : de taille n*k
# - La matrice diagonale de taille k x k avec des valeurs r�elles non-negatives sur la diagonale
# On peut faire la pr�diction en appliquant la multiplication des 3 matrices

from scipy.sparse.linalg import svds

#Obtenir les composantes de SVD � partir de la matrice User-Item du train. On choisit une valeur de k=20.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)

# Multiplication des 3 matrices avec np.dot pour obtenir la matrice User_Item estim�e.
X_pred = np.dot(np.dot(u, s_diag_matrix), vt) 

#la normalisation de X_pred vu qu'elle retourne des donn�es qui sont pas bien distribu� dans [0,5]
import math
x = np.zeros((n_users, n_items))
for i in range(0,n_items):
    a=max(X_pred[:,i])
    b=min(X_pred[:,i])
    c=0
    d=5
    for j in range(0,n_users):
        x[j,i]=(X_pred[:,i][j]-(a-c))*d/(b-a+c)
        if math.isnan(x[j,i]): x[j,i]=0

# Calcul de performance avec RMSE entre la matrice estim�e et la matrice du test
print ('RMSE: ' + str(rmse(x, test_data_matrix)))
#On a trouv� 1.4610559480936944 comme RMSE, c'est plus grand que le RMSE des mod�les Memory based, mais ca prend �normement moins du temps.

#Ce qu'on va dans la partie qui suit c'est d'am�liorer notre mod�le par le gradient stochastique et l'ALS.

#2.2 Stochastic Gradient Descent with Weighted Lambda Regularisation (SGD)
#2.2.1. La mise en place du mod�le:

# ********** Algorithme SGD (Stochastic Gradient Descent) ************
# Quand on utilise le filtrage collboratif pour SGD,on veut estimer 2 matrices U et P: 
# - La matrice U des caract�res cach�s pour les utilisateurs : de taille m*k (m: nombre d'utilisateurs, k: dimension de l'espace des caract�res cach�s)
# - La matrice P des caract�res cach�s pour les items : de taille n*k (m: nombre d'items, k: dimension de l'espace des caract�res cach�s)
# Apr�s l'estimation de U et P, on peut alors pr�dire les ratings inconnus en multipliant les matrices la transpos�e de U et P.
 
#Les matrices I et I2 serviront de matrices de s�lecteur pour prendre les �l�ments appropri�s apr�s la cr�ation du Train et du Test
#selecteur de var est �gal � 1 si la valeur dans la matrice est != 0

# matrice d'indices pour le train
I = train_data_matrix.copy()
I[I > 0] = 1
I[I == 0] = 0

# matrice d'indices pour le test
I2 = test_data_matrix.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# La fonction prediction permet de pr�dire les ratings inconnus en multipliant les matrices la transpos�e de U et P
def prediction(U,P):
    return np.dot(U.T,P)
 
# Pour mettre � jour U et P, on peut utiliser le SGD o� on it�re chaque observation dans le train pour mettre � jour U et P au fur et � mesure: 
# P_{i+1} = P_i + (gamma) (e{ui}*U_u - (lambda)* P_i)
# U_{i+1} = U_i + (gamma) (e{ui}*P_u - (lambda)* U_i)
    
# On note: 
# - (gamma) la vitesse de l'apprentissage
# - (lambda) le Terme de r�gularisation
# - e :l'erreur qui est la diff�rence entre le rating r�el et le rating pr�dit.

# ****** Initialisation ******* 
lmbda = 0.1 # Terme de r�gularisation
k = 20 # dimension de l'espace des caract�res cach�s
m, n = train_data_matrix.shape  # nombre d'utilisateurs et d'items
steps = 150  # Nombre d'it�ration 
gamma=0.001  # vitesse d'apprentissage
U = np.random.rand(k,m) # Matrice des caract�res cach�s pour les utilisateurs
P = np.random.rand(k,n) # Matrice des caract�res cach�s pour les items

#les matrices U et P sont initialis�es avec des valeurs al�atoires au d�but, mais leur contenu change � chaque it�ration en se 
#basant sur le train

#Il existe plusieurs m�triques d'�valuation, mais la plus populaire des m�triques utilis�e pour �valuer l'exactitude des ratings pr�dits
#est l'erreur quadratique moyenne (RMSE) qu'on a utilis� dans notre projet :
#RMSE =RacineCarr�e{(1/N) * sum (r_i -estim�{r_i})^2}

def rmse2(I,R,P,U):
    return np.sqrt(np.sum((I * (R - prediction(U,P)))**2)/len(R[R > 0]))

#R = train_data_matrix
#prediction(U,P): estimateur du train_data_matrix avec la m�thode de factorisation
#I pour prendre seulement la partie significative de la matrice (!=0)

#On ne consid�re que les valeurs !=0 
users,items = train_data_matrix.nonzero()  

#impl�mentation de SGD: (ps) cet algo prend du temps ca depend de nombre steps choisi.
train_errors = [] #stocker les erreurs du train obtenus par RMSE � chaque it�ration (step) 
test_errors = [] #stocker les erreurs du test obtenus par RMSE � chaque it�ration (step) 
     
for step in range(steps):
    for u, i in zip(users,items): #zip() retourne les tuples (user,item)
        e = train_data_matrix[u, i] - prediction(U[:,u],P[:,i])  # calculer l'erreur e pour le gradient
        U[:,u] += gamma * ( e * P[:,i] - lmbda * U[:,u]) # mise � jour de la matrice U
        P[:,i] += gamma * ( e * U[:,u] - lmbda * P[:,i])  # mise � jour de la matrice P
        
    train_rmse = rmse2(I,train_data_matrix,P,U) # Calcul de l'RMSE � partir du train
    test_rmse = rmse2(I2,test_data_matrix,P,U) # Calcul de l'RMSE � partir du test
    train_errors.append(train_rmse) #� chaque it�ration ajouter l'erreur � la liste
    test_errors.append(test_rmse) #� chaque it�ration ajouter l'erreur � la liste

print('RMSE : ' + str(np.mean(test_errors)))

# Maitenant, apr�s avoir obtenus toutes les valeurs de l'erreur � chaque �tape,on peut tracer la courbe d'apprentissage.
# ==> On V�rifie la performance en tracant les erreurs du train et du test

import matplotlib.pyplot as plt

plt.plot(range(steps), train_errors, marker='o', label='Training Data'); 
plt.plot(range(steps), test_errors, marker='v', label='Test Data');
plt.title('Courbe d apprentissage SGD')
plt.xlabel('Nombre d etapes');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()

# Le mod�le semble fonctionner bien avec,relativement, une basse valeur de RMSE apr�s convergence.
# La performance du mod�le peut d�pendre des param�tres (gamma), (lambda) et k qu'on a vari� � plusieurs reprises afin d'obtenir 
# le meilleur RMSE.
# Apr�s cette �tape, on peut comparer le rating r�el avec le rating estim�; Pour ce faire, on utilise la matrice User-item qu'on a 
# d�j� calcul�e et utilis� la fonction prediction(U,P) impl�ment�e pr�c�demment. 

#2.3 ALS : Alternating Least Squares

# Index matrix for training data
I = train_data_matrix.copy()
I[I > 0] = 1
I[I == 0] = 0

# Index matrix for test data
I2 = test_data_matrix.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

lmbda = 0.1 # Terme de r�gularisation
k = 20 # dimension de l'espace des caract�res cach�s
n_epochs = 2 # Nombre d'�tapes
m, n = test_data_matrix.shape # Number of users and items
U = np.random.rand(k,m) # Latent user feature matrix : # Matrice des caract�res cach�s pour les utilisateurs
P = np.random.rand(k,n) # Latent item feature matrix : # Matrice des caract�res cach�s pour les items
P[0,:] = test_data_matrix[test_data_matrix != 0].mean(axis=0) # Avg. rating for each product
#les matrices U et P sont initialis�es avec des valeurs al�atoires au d�but, mais leur contenu change � chaque it�ration en se 
#basant sur le train
E = np.eye(k) # (k x k)-dimensional idendity matrix
train_errors = []
test_errors = []

#Il existe plusieurs m�triques d'�valuation, mais la plus populaire des m�triques utilis�e pour �valuer l'exactitude des ratings pr�dits
#est l'erreur quadratique moyenne (RMSE) qu'on a utilis� dans notre projet :
#RMSE =RacineCarr�e{(1/N) * sum (r_i -estim�{r_i})^2}

def rmse2(I,R,P,U):
    return np.sqrt(np.sum((I * (R - prediction(U,P)))**2)/len(R[R > 0]))

# Repeat until convergence
for epoch in range(n_epochs):
    # Fix P and estimate U
    for i, Ii in enumerate(I):
        nui = np.count_nonzero(Ii) # Number of items user i has rated
    
        # Least squares solution
        Ai = np.dot(P, np.dot(np.diag(Ii), P.T)) + lmbda * nui * E
        Vi = np.dot(P, np.dot(np.diag(Ii), train_data_matrix[i].T))
        U[:,i] = np.linalg.solve(Ai,Vi)
        
    # Fix U and estimate P
    for j, Ij in enumerate(I.T):
        nmj = np.count_nonzero(Ij) # Number of users that rated item j
        
        # Least squares solution
        Aj = np.dot(U, np.dot(np.diag(Ij), U.T)) + lmbda * nmj * E
        Vj = np.dot(U, np.dot(np.diag(Ij), train_data_matrix[:,j]))
        P[:,j] = np.linalg.solve(Aj,Vj)
    
    train_rmse = rmse2(I,train_data_matrix,P,U)
    test_rmse = rmse2(I2,test_data_matrix,P,U)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    
    print ("[Epoch %d/%d] train error: %f, test error: %f"     %(epoch+1, n_epochs, train_rmse, test_rmse))
    
print ("Algorithm converged")

"""
Conclusion:
Cet algorithme est le meilleur de tous les autres algorithmes. 
Dans 2 it�ration on a trouv� un erreur de train qui est �gale � 0.773646 et un erreur de test qui est �gale � 1.273079
Comme c'est l'algorithme le plus rapide et le plus efficace, On a d�cid� de le g�n�ralis� sur tout le jeu de donn�es.
"""