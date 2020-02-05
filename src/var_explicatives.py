import numpy as np
from conllu import parse_incr


def richess_lexicale(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  cpt = 0
  taille = 0
  liste =[]

  for i in range (len(sequences)):
    sentence = sequences[i]    
    taille += len(sequences[i])
    for j in range(len(sequences[i])):  
      token = sentence[j]
      liste.append(token['form'])
  return float(len(set(liste))/taille)
  
  

def part_of_adp_case(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  cpt = 0
  taille = 0

  for i in range (len(sequences)):
    sentence = sequences[i]    
    taille += len(sequences[i])
    for j in range(len(sequences[i])):  
      token = sentence[j]
      # print(token)
      if token['upostag'] == 'ADP' and token['deprel'] == 'case':
        cpt += 1
  
  return float(cpt/taille)


def mean_length_dependencies(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None and token['head'] != 0 and  (token['upostag'] == 'SCONJ'):
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)
    
    

def nb_sentences(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  cpt = 0
  taille = 0
  taille2 = 0

  taille2 = len(sequences)
  for i in range (len(sequences)):
    sentence = sequences[i]    
    taille += len(sequences[i])
    for j in range(len(sequences[i])): 
      # taille2 += len(sequences[j]) 
      token = sentence[j]
 
  return taille2
  
  

def dep_noun(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None and token['head'] != 0 and  (token['upostag'] == 'NOUN') :
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)
    
    
def dep_verb(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None and token['head'] != 0 and  (token['upostag'] == 'VERB') :
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)
    
    

def mean_dependencies_case(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None  and  (token['deprel'] == 'case'):
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)
    

def mean_dependencies_amod(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None  and  (token['deprel'] == 'amod'):
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)
    

def mean_dependencies_flat(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None  and  (token['deprel'] == 'flat'):  #obj #clf  #mark  #cc 
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)
    

def mean_dependencies_obj(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None  and  (token['deprel'] == 'obj'):   #clf  #mark  #cc 
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)


def mean_dependencies_clf(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None  and  (token['deprel'] == 'clf'):    #mark  #cc 
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)


def mean_dependencies_mark(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None  and  (token['deprel'] == 'mark'):     #cc 
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)


def mean_dependencies_cc(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None  and  (token['deprel'] == 'cc'):     
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)
    

def tx_non_proj(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  cpt = 0
  non_proj = 0
  taille = 0
  taille2 = 0
  # mot1 = []
  # mot2 = []

  taille2 = len(sequences)
  for i in range (len(sequences)):
    sentence = sequences[i]    
    taille += len(sequences[i])
    mot1 = []
    mot2 = []
    for j in range(len(sequences[i])): 
      # taille2 += len(sequences[j]) 
      token = sentence[j]
      if token['head'] != None:
        mot1.append(token['id'])
        mot2.append(token['head'])
      # print(len(mot1))
    for k in range (len(mot1)):
      for l in range (1,len(mot1)):
        if mot2[l] > mot1[k] and mot2[l] < mot2[k]:
          non_proj += 1
          break

  # return non_proj
  return float(non_proj/taille)
  

def nb_mots_par_phrase(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  cpt = 0
  taille = 0
  taille2 = 0
  mot1 = []
  # mot2 = []

  taille2 = len(sequences)
  for i in range (len(sequences)):
    sentence = sequences[i]    
    taille += len(sequences[i])
    mot1.append(len(sentence))
    # for j in range(len(sequences[i])): 
    #   # taille2 += len(sequences[j]) 
    #   token = sentence[j]
      # cpt += len(token)
      # mot1.append(len(token['lemma']))
  return np.mean(mot1)
  # return float(cpt/taille)
  
  
def richesse_morpho(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  cpt = 0
  taille = 0
  liste =[]
  total = 0

  for i in range (len(sequences)):
    sentence = sequences[i]    
    taille += len(sequences[i])
    for j in range(len(sequences[i])):  
      token = sentence[j]
      if token['feats'] != None:
        total += len(token['feats'])
  return float(total/taille)
  
  
def nb_noun(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None and token['head'] != 0 and  (token['upostag'] == 'ADJ'): 
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)


def dep_adv(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None and token['head'] != 0 and  (token['upostag'] == 'ADV'): 
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)
    
    
def dep_det(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  length_dependencies = []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None and token['head'] != 0 and  (token['upostag'] == 'DET'):  #DET CCONJ
        length_dependencies.append(np.abs(token['head'] - token['id']))
  if not length_dependencies:
    return 0
  else:
    return np.mean(length_dependencies)



def dep_discourse(train):
  sequences = []
  for tokenlist in parse_incr(train):  
    sequences.append(tokenlist)  
  lis= []

  for i in range (len(sequences)):
    sentence = sequences[i]
    for j in range(len(sequences[i])):
      token = sentence[j]
      # print(token)
      if token['head'] != None and  (token['deprel'] == 'discourse'): 
        lis.append(len(token))
  if not lis:
    return 0
  else:
    return np.mean(lis)
