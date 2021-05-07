
import pandas as pd
from nltk.corpus import stopwords
import string
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


#___________________________________________________________ Fonctions utiles ___________________________________________________________

print("Enter file name (sans .csv) : ")
file_name = input()

stop_words =stopwords.words('french')
stop_words.extend(['prenom','alors', 'au', 'aucuns', 'aussi', 'autre', 'avant', 'avec', 'avoir', 'car', 'ce', 'cela', 'ces', 'ceux', 'chaque', 'ci', 'comme', 'comment', 'dans', 'des', 'du', 'dedans', 'dehors', 'depuis', 'devrait', 'doit', 'donc', 'dos', 'dÃ©but', 'elle', 'elles', 'en', 'encore', 'essai', 'est', 'et', 'eu', 'fait', 'faites', 'fois', 'font', 'hors', 'ici', 'il', 'ils', 'je', 'la', 'le', 'les', 'leur', 'lÃ\xa0', 'ma', 'maintenant', 'mais', 'mes', 'mien', 'mon', 'mot', 'mÃªme', 'ni', 'nommÃ©s', 'notre', 'nous', 'ou', 'oÃ¹', 'par', 'parce', 'pas', 'peut', 'peu', 'plupart', 'pour', 'pourquoi', 'quand', 'que', 'quel', 'quelle', 'quelles', 'quels', 'qui', 'sa', 'sans', 'ses', 'seulement', 'si', 'sien', 'son', 'sont', 'sous', 'soyez', 'sujet', 'sur', 'ta', 'tandis', 'tellement', 'tels', 'tes', 'ton', 'tous', 'tout', 'trop', 'trÃ¨s', 'tu','t','j','J','je','Je','qu','jusqu','l','L','À','à', 'voient', 'vont', 'votre', 'vous' ,'vu', 'Ã§a', 'Ã©taient', 'Ã©tat', 'Ã©tions', 'Ã©tÃ©', 'Ãªtre'])


def replace_ponctuation(s):
    if s in string.punctuation :
        return ' '
    return s

def clean_stop_words(string):
    string_clean =""
    string = "".join([replace_ponctuation(s) for s in string])
    for word in string.split():
        if not word in stop_words:
            string_clean += word + " "
    return string_clean

stemmer = FrenchStemmer()
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(stemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)
#___________________________________________________________ Read file as dataframe ___________________________________________________________

df = pd.read_csv (file_name+'.csv')


columns_list = list(df.columns)
rows_list=[]
for i in range(len(columns_list)):
    new_row=[]
    for index,row in df.iterrows():
        if(type(row[i])) == str :
            new_row.append(stemSentence(clean_stop_words(row[i])))
        else :
            new_row.append(row[i])
    rows_list.append(new_row)


#___________________________________________________________ write csv file from the new dataframe ___________________________________________________________


zipObj = zip(columns_list,rows_list)

d = dict(zipObj)

dfnew = pd.DataFrame(data=d)


dfnew.to_csv(file_name+'_CLEAN.csv',index = False)

#___________________________________________________________ liste de liste des mots ___________________________________________________________
f = open(file_name+".txt", "w")

phrases = []
for L in rows_list:
    for phrase in L:
        phrases.append(phrase)

for phrase in phrases:
    if (type(phrase) == str):
        f.write(phrase+"\n")
f.close()





