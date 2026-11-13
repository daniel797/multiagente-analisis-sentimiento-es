import sys,numpy as np,pandas as pd
sys.path[:0]=['scripts/experimento']
import validacion_cruzada as VC, optimizar_hiperparametros as H
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from scipy.sparse import hstack,csr_matrix
from sklearn.preprocessing import StandardScaler
train,dev=H.cargar('data'); VC.Representacion.precalcular(train,dev)
y=train.etiqueta.to_numpy(); cols=VC.F._numericas(VC.F.GRUPOS_NUM)
T=train.texto.map(lambda t:VC.V.aplicar(t,'base'))
def cv(nombre,fn):
    f=[]
    for tr,va in VC.pliegues(train):
        p=fn(tr,va); f.append(f1_score(y[va],p,average='macro'))
    print(nombre,round(np.mean(f),4),round(np.std(f,ddof=1),4),[round(x,3) for x in f],flush=True)
def svm(tr,va):
    v=VC._vec_palabras().fit(T.iloc[tr]); return LinearSVC(C=0.3,class_weight='balanced').fit(v.transform(T.iloc[tr]),y[tr]).predict(v.transform(T.iloc[va]))
def nb(tr,va):
    v=VC._vec_palabras().fit(T.iloc[tr]); return ComplementNB(alpha=0.5).fit(v.transform(T.iloc[tr]),y[tr]).predict(v.transform(T.iloc[va]))
def hgb(tr,va):
    return HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=200,class_weight='balanced',random_state=0).fit(train.iloc[tr][cols],y[tr]).predict(train.iloc[va][cols])
def charvars(tr,va):
    v=VC._vec_caracteres().fit(T.iloc[tr]); sc=StandardScaler().fit(train.iloc[tr][cols])
    X=lambda i:hstack([v.transform(T.iloc[i]),csr_matrix(sc.transform(train.iloc[i][cols]))]).tocsr()
    return VC._lr().fit(X(tr),y[tr]).predict(X(va))
def charsvm(tr,va):
    v=VC._vec_caracteres().fit(T.iloc[tr]); return LinearSVC(C=0.1,class_weight='balanced').fit(v.transform(T.iloc[tr]),y[tr]).predict(v.transform(T.iloc[va]))
for n,f in [('svm_palabras',svm),('nb_palabras',nb),('hgb_24vars',hgb),('char+24vars',charvars),('svm_char',charsvm)]: cv(n,f)
