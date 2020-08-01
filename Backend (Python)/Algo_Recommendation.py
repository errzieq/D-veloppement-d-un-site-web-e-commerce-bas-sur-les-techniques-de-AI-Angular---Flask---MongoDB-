#Nettoyage des données:
import pandas as pd 
import numpy as np
import random
random.seed(9001) #pour ne pas avoir toujours les memes erreurs à chaque fois qu'on re exécute le projet.
import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client[ "PFE" ]

#products
products = db[ "products" ]
products_df = pd.DataFrame(list(products.find()))
products_df = products_df.drop('_id', 1)

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

#Un des problémes qu'on a rencontré était le fait que les ids des produits n'était pas ordonnée.
#C'est à dire on peut trouver l'utilisateur 1,2,3,5 et 8 sans trouver les utilisateurs 4 ,6 et 7.
#Ceci nous a posé un probléme dans la création de la matrice user-produit parce que on risque d'avoir plusieurs lignes et colonnes vides.
#Pour ca, on a crée un tableau indice_user et un tableau indice_produit qui contiennent les anciens id et les nouvelles id par expl (1,2,5,6)=>(1,2,3,4) 
#puis on a ajouté deux colonnes sur le tableau principale qui contient ces nouveaux IDs.
#on a exporté les nouvelles ids dans un fichier csv, et à chaque fois on utilise ce nouveau fichier.

indice_user = pd.DataFrame()
indice_user["indice"]=range(1,len(useri)+1)
indice_user["useri"]=useri

indice_item = pd.DataFrame()
indice_item["indice"]=range(1,len(itemi)+1)
indice_item["itemi"]=itemi
 
#créer user_ID_new et Item_ID_new
x=[]
y=[]

for i in range(0,len(tab)):
    x.append((indice_user.indice[indice_user.useri==tab.userId.iloc[i]].axes[0]+1)[0])
    y.append((indice_item.indice[indice_item.itemi==tab.produitId.iloc[i]].axes[0]+1)[0])

tab["userIdnew"]=x
tab["produitIdnew"]=y

tab[:20]

data_matrix = np.zeros((n_users, n_items))
for line in tab[["userIdnew","produitIdnew","rating"]].itertuples():#parcourir la ligne col par col
    data_matrix[line[1]-1, line[2]-1] = line[3] 
    
#functions to pass from old to new Id
def change_item_to_newId(id):
    return tab.produitIdnew[tab.produitId==id].iloc[0]
def change_item_to_Id(id):
    return tab.produitId[tab.produitIdnew==id].iloc[0]

def change_user_to_newIduser(id):
    return tab.userIdnew[tab.userId==id].iloc[0]
def change_user_to_Iduser(id):
    return tab.userId[tab.userIdnew==id].iloc[0]

#Partie1 ALS : Alternating Least Squares
    
# Index matrix for training data
# I = Indice si un produit a une réaction par un utilisateur (1: oui , 0: non)
I = data_matrix.copy()
I[I > 0] = 1
I[I == 0] = 0

lmbda = 0.1 # Terme de régularisation
k = 20 # dimension de l'espace des caractères cachés
n_epochs = 2 # Nombre d'étapes
m, n = data_matrix.shape # Number of users and items
U = np.random.rand(k,m) # Latent user feature matrix : # Matrice des caractères cachés pour les utilisateurs
P = np.random.rand(k,n) # Latent item feature matrix : # Matrice des caractères cachés pour les items
P[0,:] = data_matrix[data_matrix != 0].mean(axis=0) # Avg. rating for each product
#les matrices U et P sont initialisées avec des valeurs aléatoires au début, mais leur contenu change à chaque itération en se 
#basant sur le train
E = np.eye(k) # (k x k)-dimensional idendity matrix

#Il existe plusieurs métriques d'évaluation, mais la plus populaire des métriques utilisée pour évaluer l'exactitude des ratings prédits
#est l'erreur quadratique moyenne (RMSE) qu'on a utilisé dans notre projet :
#RMSE =RacineCarrée{(1/N) * sum (r_i -estimé{r_i})^2}
def rmse2(I,R,P,U):
    return np.sqrt(np.sum((I * (R - prediction(U,P)))**2)/len(R[R > 0]))

# La fonction prediction permet de prédire les ratings inconnus en multipliant les matrices la transposée de U et P
def prediction(U,P):
    return np.dot(U.T,P)

train_errors = []
test_errors = []

# Repeat until convergence
for epoch in range(n_epochs):
    # Fix P and estimate U
    for i, Ii in enumerate(I):
        nui = np.count_nonzero(Ii) # Number of items user i has rated
        
        # Least squares solution
        Ai = np.dot(P, np.dot(np.diag(Ii), P.T)) + lmbda * nui * E
        Vi = np.dot(P, np.dot(np.diag(Ii), data_matrix[i].T))
        U[:,i] = np.linalg.solve(Ai,Vi)
        
    # Fix U and estimate P
    for j, Ij in enumerate(I.T):
        nmj = np.count_nonzero(Ij) # Number of users that rated item j
        
        # Least squares solution
        Aj = np.dot(U, np.dot(np.diag(Ij), U.T)) + lmbda * nmj * E
        Vj = np.dot(U, np.dot(np.diag(Ij), data_matrix[:,j]))
        P[:,j] = np.linalg.solve(Aj,Vj)
    
    train_rmse = rmse2(I,data_matrix,P,U)
    train_errors.append(train_rmse)
    
    
    print ("[Epoch %d/%d] train error: %f" \
    %(epoch+1, n_epochs, train_rmse))
    
print ("Algorithm converged")

model_matrix=prediction(U,P) #matrice (user,item)
model_matrix[model_matrix < 0.5] = 0
model_matrix[(model_matrix >= 0.5) & (model_matrix < 1.5)] = 1
model_matrix[(model_matrix >= 1.5) & (model_matrix < 2.5)] = 2
model_matrix[(model_matrix >= 2.5) & (model_matrix < 3.5)] = 3
model_matrix[(model_matrix >= 3.5) & (model_matrix < 4.5)] = 4
model_matrix[model_matrix >= 4.5] = 5
model_df=pd.DataFrame(model_matrix)

model_df

def als_recom_it_for_user(id):
    id=id-1
    similar_indices = model_matrix[id].argsort()[:-250:-1]
    #model_matrix[id] les rates pour un utilisateur 
    #.argsort()[:-250:-1] ordonnée et on prend les 250 les plus grande
    similar_items = [(model_matrix[id][i], change_item_to_Id(i+1)) for i in similar_indices]
    #ps: i+1 pour éliminer l'effet du faite qu'on commance avec un zéro notre liste des items!
    return similar_items[:]

recommended = []
for i in range(0,n_users):
    
    recommand = als_recom_it_for_user(i)
    df = pd.DataFrame(recommand, columns =['rating', 'produitId']) 
    rv = df.to_json(orient='records')
    recommended.append(rv)
    
resultat = pd.DataFrame(recommended)

resultat['userId'] = resultat.index+1

for i in range(0,n_users-1):
    resultat["userId"][i] = change_user_to_Iduser(i+1)
    
resultat.to_csv(r'C:/Users/errza/OneDrive/Bureau/PFE_FILES_FIN/ALSRatings.csv')