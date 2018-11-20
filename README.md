# Face-Recognition-via-machine-learning
Face recognition using knn
Le classificateur K le plus proche voisin (knn) est une extension du système de classification du plus proche voisin (NN). Le classificateur le plus proche fonctionne sur la base d'une décision non paramétrique simple. Chaque image de requête Iq est examinée en fonction de la distance de ses caractéristiques par rapport aux caractéristiques d'autres images dans la base de données d'apprentissage. Le voisin le plus proche est l'image qui a la distance minimale de l'image de la requête dans l'espace de caractéristiques. La distance entre deux entités peut être mesurée sur la base de l'une des fonctions de distance telles que la distance de bloc de ville d1, la distance euclidienne d2 ou la distance de cosinus dcos: 
  
PS : Dans notre cas on va utiliser la distance euclidienne d2.
Implémentation de l’algorithme :
#importation des modules:
import numpy as np
import cv2
# instancier l'objet camera pour capturer les visages
cam = cv2.VideoCapture(0)
# creer un haarcascade pour la detection faciale
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

# creer un placeholder pour le stockage des données
data = []
ix = 0	# le numéro de frame courante
while True:
	# valeur booleene pour la l'approvisionnement de l'objet cam
	ret, fr = cam.read()
	# si la caméra fonctionne bien, nous procédons à l'extraction du visage
	if ret == True:
		# convertir la frame courante en grayscale
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		# appliquer la haarcascade pour detecter visages dans la frame courante
		# les autres parametres 1.3 and 5 sont des tuning parameteres pour
                             haarcascade objet
		faces = facec.detectMultiScale(gray, 1.3, 5)
		for (x, y, w, h) in faces:
			# recupere le composant visage à partir de la frame
			fc = fr [y:y+h, x:x+w, :]
			# redimensionner l'image de visage en 50X50X3
			r= cv2.resize(fc, (50, 50))
			# stocker les données de visage après chaque 10 images
			# seulement si le nombre d'entrées est inférieur à 40
			if ix%10 == 0 and len(data) < 20:
				data.append(r)
			# pour la visualisation, dessinez un rectangle autour du visage
			cv2.rectangle(fr, (x, y), (x+w, y+h), (0, 255, 0), 2)
		ix += 1	# increment le numéro de frame courante
		cv2.imshow('frame', fr)	# afficher le cadre
                          	# si le temps d’attente atteint 27seconde(pour bien simuler le visage)
                                       #ou le nombre d'images frappe 20, nous nous arrêtons
		         # enregistrement.
		if cv2.waitKey(1) == 27 or len(data) >= 20:
			break
	else:
		         # si la caméra ne fonctionne pas, imprimez "erreur"

		print "error"

cv2.destroyAllWindows()
# convertir les données en un format numérique
data = np.asarray(data)
# imprimer la forme comme une vérification de la santé
print data.shape
# enregistrer les données sous forme de matrice chiffrée dans un format codé, un fich de format npy est généré dans le même dossier.
np.save('face_01', data)
# Nous allons exécuter le script pour différentes personnes et stocker les données dans    plusieurs fichiers
#génération du fichier codé 'face_01'

On adopte le code de l’algorithme (knn) :
import numpy as np
from matplotlib import pyplot as plt
# module pyplot de la biblio matplotlib permet de faire des graphes


mean_01 = np.asarray([0., 2.]) #convertir les données en un format numérique
sigma_01 = np.asarray([[1.0, 0.0], [0.0, 1.0]])#créer la matrice identité d'ordre 2

mean_02 = np.asarray([4., 0.])
sigma_02 = np.asarray([[1.0, 0.0], [0.0, 1.0]])

print mean_01      #[0. 2.]
print sigma_01     
                   #[ [1. 0.]
                   #  [0. 1.] ]
data_01 = np.random.multivariate_normal(mean_01, sigma_01, 500)
#parametres(Moyenne de la distribution N-dimensionnelle,Matrice de covariance
#de la distribution,size)
data_02 = np.random.multivariate_normal(mean_02, sigma_02, 500)
#La distribution multivariée normale, multinormale ou gaussienne est
#une généralisation de la distribution normale unidimensionnelle aux
#dimensions supérieures. Une telle distribution est spécifiée par sa matrice
#de moyenne et de covariance. Ces paramètres sont analogues à la moyenne
#(moyenne ou «centre») et à la variance (écart-type ou «largeur» au carré) de
#la distribution normale unidimensionnelle. Chaque entrée [i, j, ...,:] est une
#valeur N-dimensionnelle tirée de la distribution.
print data_01.shape,data_02.shape #imprimer la forme comme une vérification de la santé
#==>(500L, 2L) (500L, 2L)
plt.figure(0) #parametre num=0 pour referer que le titre de la fenêtre sera réglé sur
              #le num de cette figure
plt.xlim(-4,10)#Obtenir ou définir les limites x des axes actuels
plt.ylim(-4,6) #Obtenir ou définir les limites y des axes actuels
plt.grid('on') #Activer les grilles d'axes
plt.scatter(data_01[:, 0], data_01[:, 1], color='red')#Un diagramme de dispersion de
#y contre x avec une taille de marqueur et / ou une couleur variable.
plt.scatter(data_02[:, 0], data_02[:, 1], color='green')
plt.show()#projection du graphe


labels = np.zeros((1000,1))#Renvoie un nouveau tableau de forme et de type donné,
                           #rempli de zéros.
labels[500:,:] = 1.0#remplir la moitié de tableau par 1

data = np.concatenate([data_01,data_02],axis = 0)#np.concatenate permet de Jogner
#une séquence de tableaux le long d'un axe existant
print data.shape

ind = range(1000)#generer la liste [0 1 ..999]
np.random.shuffle(ind)#Modifier la séquence 'ind' sur place en mélangeant son contenu

print ind[:10]
#melange
data = data[ind]
labels = labels[ind]
print data.shape,labels.shape #traçage data et labels

def distance(x1,x2):
	return np.sqrt(((x1-x2)**2).sum())#definir la distance euclédienne pour
                                          #referer les data set aux outpout

def knn(x, train, targets, k = 5):
	m = train.shape[0]
	dist = []
	for ix in range(m):
		pass #instruction est requise de manière syntaxique mais pas exécutable
		dist.append(distance(x,train[ix]))
	dist = np.asarray(dist) #convertir en tableau numpy
	indx = np.argsort(dist) #Renvoie les indices qui trieraient le tableau dist
	# dist[indx] sera trié
	#  print labels[indx] # max of this is answer
	sorted_labels = labels[indx][:k] 
	# print sorted_labels

	#liste des valeurs uniques et leur nombre
	counts = np.unique(sorted_labels,return_counts = True)#Trouvez les éléments
	#uniques d'un tableau,ainsi les indices du tableau d'entrée qui donnent les
	#valeurs uniques
	return counts[0][np.argmax(counts[1])]#Renvoie les indices des valeurs maximales
                                              #le long d'axe.
	# unique nos : count of nos
x_test = np.asarray([2.0, 0.0])
knn(x_test,data,labels) 

#accuracy
split = int(data.shape[0]*0.75)

X_train = data[:split]
X_test = data[split:]
Y_train = labels[:split]
Y_test = labels[split:]
print X_train.shape,X_test.shape
print Y_train.shape,Y_test.shape
preds = []
#calcul pour 250 vecteurs test
for tx in range(X_test.shape[0]):
    preds.append(knn(X_test[tx], X_train, Y_train))
preds = np.asarray(preds).reshape((250, 1))
print preds.shape

print 100*(preds == Y_test).sum()/float(preds.shape[0])
Après implementation, l’algorithme affiche un pourcentage de comparaison entre les matrices numpy mean et sigma, avec un graphe qui modélise la répartition gaussienne de chaque états avec différents couleurs (rouge et vert)

Le programme de reconnaissance faciale sera :
import numpy as np
import cv2
# instantiate the camera object and haar cascade
cam = cv2.VideoCapture(1)
facec = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
# declare the type of font to be used on output window
font = cv2.FONT_HERSHEY_SIMPLEX
# load the data from the numpy matrices and convert to linear vectors 
f_01 = np.load('face_01.npy').reshape((40, -1))     #Mouad
f_02 = np.load('face_02.npy').reshape((40, -1))     #Oussama
f_03 = np.load('face_03.npy').reshape((20, -1))     #Walid
# create a look-up dictionary
names = {
	0: 'Mouad',
	1: 'Oussama', 
	2: 'Walid',} 
# combine all info into one data array
data = np.concatenate((f_01, f_02, f_03))	# (120, -1)
# create a matrix to store the labels
labels = np.zeros((data.shape[0]))
labels[40:80] = 1.0
labels[80:] = 2.0
 #Define KNN functions
def distance(x1, x2):
	d = np.sqrt(((x1 - x2) ** 2).sum())
	return d 


def knn(X_train, y_train, xt, k = 10):
	vals = []
	for ix in range(X_train.shape[0]):
		d = distance(X_train[ix], xt)
# compute distance from each point and store in dist
		vals.append([d, y_train[ix]])
	sorted_labels = sorted(vals, key = lambda z:z[0])
	neighbours = np.asarray(sorted_labels)[:k, -1]
	freq = np.unique(neighbours, return_counts = True)
	return freq[0][freq[1].argmax()]

#Run the main loop 
while True:
            # get each frame
	ret, fr = cam.read()
	if ret == True:
                         # convert to grayscale and get faces
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		faces = facec.detectMultiScale(gray, 1.3, 5)
		for (x, y, w, h) in faces:
			#Extract detected face
			fc = fr[y:y+h, x:x+w, :]
			#Resize to a fixed shape
                                       # after processing the image and rescaling, convert to linear vector   using .flatten()  and pass to knn function along with all the data.
			r = cv2.resize(fr, (50, 50)).flatten()
                                       # display the name
			text = names[int(knn(data, labels, r))]
			cv2.putText(fr, text, (x, y), font, 1, (0, 0, 255), 2)

                                       # draw a rectangle over the face
			cv2.rectangle(fr, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.imshow('reconnaissance faciale', fr)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		print "error"
		break
cv2.destroyAllWindows() 


Le principal avantage d'une telle approche basée sur la mémoire est que le classificateur s'adapte immédiatement à mesure que nous recueillons de nouvelles données. Toutefois, l'inconvénient est que la complexité de calcul pour classer les nouveaux échantillons croît linéairement avec le nombre d'échantillons dans le DataSet dans le pire des scénarios, sauf si le DataSet a très peu de dimensions (fonctionnalités) et l'algorithme a été mis en œuvre en utilisant des structures de données efficaces.
