from utils import *
from var_explicatives import *


variables = []
variables.append(var(lexical_richness))
variables.append(var(part_of_adp_case))
variables.append(var(mean_length_dependencies))
variables.append(var(nb_sentences))
variables.append(var(dep_noun))
variables.append(var(dep_verb))
variables.append(var(mean_dependencies_case))
variables.append(var(mean_dependencies_amod))
variables.append(var(mean_dependencies_flat))
variables.append(var(mean_dependencies_obj))
variables.append(var(mean_dependencies_clf))
variables.append(var(mean_dependencies_mark))
variables.append(var(mean_dependencies_cc))
variables.append(var(non_projectivity_rate))
variables.append(var(words_per_sentences))
variables.append(var(morphological_richness))
variables.append(var(nb_noun))
variables.append(var(dep_adv))
variables.append(var(dep_det))
variables.append(var(dep_discourse))



X = np.empty((len(variables),len(langues)))
y_pred = np.empty((len(variables),len(langues)))
scores = np.empty(len(variables))

for i in range (len(variables)):
	X[i], y_pred[i], scores[i] = regr(variables[i])


X_mult = X.T.reshape((len(langues),len(variables)))
print('*'*50)
print('X :',X)

print('*'*50)
print('Score régression linéaire simple :',scores)


score_regr_mult = regr_mult(X_mult)
print('*'*50)
print('Score régression linéaire multiple :', score_regr_mult)

#affichage d'une régression linéaire
#for i in range (2):
	#plot(X[i], y_pred[i], i) 
