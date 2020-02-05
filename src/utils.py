import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from conllu import parse_incr
from io import open
from matplotlib.pyplot import figure


y = np.array([64.28, 73.40, 72.06, 63.84, 68.12, 72.55, 67.18, 66.93, 64.76, 58.80, 65.22,62.77, 71.36, 65.13, 64.68,79.47, 63.58,62.73, 
			  67.05, 78.38, 75.74, 53.12, 62.30,57.44, 73.25, 76.18, 70.73, 65.13, 69.70,63.47, 71.10, 47.28, 65.85,76.33,59.77,59.91])

langues = ['ar','bg','ca','cs','da','el','en','es','et','eu','fa','fi','fr','ga','he','hi','hr','hu','id','it','ja','ko','lv','nl','no','pl','pt','ro','ru','sl','sv','tr','uk','ur','vi','zh']

nom_variables = ['lexical_richness', 'part_of_adp_case', 'mean_length_dependencies', 'nb_sentences', 'dep_noun', 'dep_verb', 'mean_dependencies_case', 
				 'mean_dependencies_amod', 'mean_dependencies_flat', 'mean_dependencies_obj', 'mean_dependencies_clf', 'mean_dependencies_mark', 
				 'mean_dependencies_cc', 'non_projectivity_rate', 'words_per_sentences', 'morphological_richness', 'nb_noun', 'dep_adv', 'dep_det', 'dep_discourse']

def open_files():
  ar_train = open("corpus_equilibre/ar/ar_train.conllu", "r", encoding="utf-8")
  bg_train = open("corpus_equilibre/bg/bg_train.conllu", "r", encoding="utf-8")
  ca_train = open("corpus_equilibre/ca/ca_train.conllu", "r", encoding="utf-8")
  cs_train = open("corpus_equilibre/cs/cs_train.conllu", "r", encoding="utf-8")
  da_train = open("corpus_equilibre/da/da_train.conllu", "r", encoding="utf-8")
  el_train = open("corpus_equilibre/el/el_train.conllu", "r", encoding="utf-8")
  en_train = open("corpus_equilibre/en/en_train.conllu", "r", encoding="utf-8")
  es_train = open("corpus_equilibre/es/es_train.conllu", "r", encoding="utf-8")
  et_train = open("corpus_equilibre/et/et_train.conllu", "r", encoding="utf-8")
  eu_train = open("corpus_equilibre/eu/eu_train.conllu", "r", encoding="utf-8")
  fa_train = open("corpus_equilibre/fa/fa_train.conllu", "r", encoding="utf-8")
  fi_train = open("corpus_equilibre/fi/fi_train.conllu", "r", encoding="utf-8")
  fr_train = open("corpus_equilibre/fr/fr_train.conllu", "r", encoding="utf-8")
  ga_train = open("corpus_equilibre/ga/ga_train.conllu", "r", encoding="utf-8")
  he_train = open("corpus_equilibre/he/he_train.conllu", "r", encoding="utf-8")
  hi_train = open("corpus_equilibre/hi/hi_train.conllu", "r", encoding="utf-8")
  hr_train = open("corpus_equilibre/hr/hr_train.conllu", "r", encoding="utf-8")
  hu_train = open("corpus_equilibre/hu/hu_train.conllu", "r", encoding="utf-8")
  id_train = open("corpus_equilibre/id/id_train.conllu", "r", encoding="utf-8")
  it_train = open("corpus_equilibre/it/it_train.conllu", "r", encoding="utf-8")
  ja_train = open("corpus_equilibre/ja/ja_train.conllu", "r", encoding="utf-8")
  ko_train = open("corpus_equilibre/ko/ko_train.conllu", "r", encoding="utf-8")
  lv_train = open("corpus_equilibre/lv/lv_train.conllu", "r", encoding="utf-8")
  nl_train = open("corpus_equilibre/nl/nl_train.conllu", "r", encoding="utf-8")
  no_train = open("corpus_equilibre/no/no_train.conllu", "r", encoding="utf-8")
  pl_train = open("corpus_equilibre/pl/pl_train.conllu", "r", encoding="utf-8")
  pt_train = open("corpus_equilibre/pt/pt_train.conllu", "r", encoding="utf-8")
  ro_train = open("corpus_equilibre/ro/ro_train.conllu", "r", encoding="utf-8")
  ru_train = open("corpus_equilibre/ru/ru_train.conllu", "r", encoding="utf-8")
  sl_train = open("corpus_equilibre/sl/sl_train.conllu", "r", encoding="utf-8")
  sv_train = open("corpus_equilibre/sv/sv_train.conllu", "r", encoding="utf-8")
  tr_train = open("corpus_equilibre/tr/tr_train.conllu", "r", encoding="utf-8")
  uk_train = open("corpus_equilibre/uk/uk_train.conllu", "r", encoding="utf-8")
  ur_train = open("corpus_equilibre/ur/ur_train.conllu", "r", encoding="utf-8")
  vi_train = open("corpus_equilibre/vi/vi_train.conllu", "r", encoding="utf-8")
  zh_train = open("corpus_equilibre/zh/zh_train.conllu", "r", encoding="utf-8")

  return ar_train,bg_train ,ca_train ,cs_train ,da_train ,el_train ,en_train ,es_train ,et_train ,eu_train ,fa_train ,fi_train ,fr_train,ga_train ,he_train ,hi_train ,hr_train ,hu_train,id_train ,it_train,ja_train ,ko_train ,lv_train ,nl_train  ,no_train, pl_train,pt_train  ,ro_train  ,ru_train  ,sl_train ,sv_train ,tr_train ,uk_train ,ur_train ,vi_train ,zh_train 


def var(name_funct):
  lg_train = open_files()
  var = []
  for train in lg_train:
    var.append(name_funct(train))
  return var


def regr(var):
  X = np.asarray(var)

  X_reg = X.reshape(-1,1)
  regressor = LinearRegression()
  regressor.fit(X_reg, y)
  y_pred = regressor.predict(X_reg)  # make predictions
  return X, y_pred, r2_score(y, y_pred, multioutput='uniform_average')


def regr_mult(X):
  regressor = LinearRegression()
  regressor.fit(X, y)
  y_pred = regressor.predict(X)  # make predictions
  score = r2_score(y, y_pred, multioutput='uniform_average')
  
  return score
  

def plot(X, y_pred, i):
  figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
  # Visualiser les résultats
  #plt.scatter(X_test, y_test, color = 'red')
  plt.scatter(X, y, color = 'blue')
  plt.plot(X, y_pred, color = 'red')
  plt.title('{} (Var {}) related to score LAS'.format(nom_variables[i],i))
  plt.xlabel('{} (Var {})'.format(nom_variables[i],i))
  plt.ylabel('LAS')
  for i in range (len(X)):
    plt.text(X[i],y[i]+0.5,langues[i])
  plt.show()

