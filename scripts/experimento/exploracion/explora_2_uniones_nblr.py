import sys,numpy as np,pandas as pd,time
sys.path[:0]=['scripts/experimento']
import validacion_cruzada as VC, optimizar_hiperparametros as H
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score
from scipy.sparse import hstack,csr_matrix
from sklearn.preprocessing import StandardScaler
train,dev=H.cargar('data'); VC.Representacion.precalcular(train,dev)
y=train.etiqueta.to_numpy(); cols=VC.F._numericas(VC.F.GRUPOS_NUM)
T=train.texto.map(lambda t:VC.V.aplicar(t,'base'))
CL=['NEG','NEU','POS']
def cv(nombre,fn):
    t=time.time();f=[]
    for tr,va in VC.pliegues(train):
        f.append(f1_score(y[va],fn(tr,va),average='macro'))
    print(nombre,round(np.mean(f),4),round(np.std(f,ddof=1),4),[round(x,3) for x in f],f'{time.time()-t:.0f}s',flush=True)
def num(tr,va):
    sc=StandardScaler().fit(train.iloc[tr][cols]); return (lambda i:csr_matrix(sc.transform(train.iloc[i][cols])))
def union(C=1,vars_=True,w=1.0):
    def fn(tr,va):
        vw=VC._vec_palabras().fit(T.iloc[tr]); vc=VC._vec_caracteres().fit(T.iloc[tr]); N=num(tr,va)
        X=lambda i:hstack([vw.transform(T.iloc[i]),vc.transform(T.iloc[i])]+([N(i)*w] if vars_ else [])).tocsr()
        return VC._lr(C).fit(X(tr),y[tr]).predict(X(va))
    return fn
def nbsvm(vars_=True,C=1,beta=1.0):
    def fn(tr,va):
        cvz=CountVectorizer(ngram_range=(1,2),min_df=2,binary=True).fit(T.iloc[tr]); Xtr=cvz.transform(T.iloc[tr]);Xva=cvz.transform(T.iloc[va]); N=num(tr,va)
        P=[]
        for k in CL:
            yk=(y[tr]==k); p=np.asarray(Xtr[yk].sum(0)).ravel()+1; q=np.asarray(Xtr[~yk].sum(0)).ravel()+1
            r=np.log((p/p.sum())/(q/q.sum()))
            A=lambda X,i:hstack([X.multiply(r).tocsr()]+([N(i)] if vars_ else [])).tocsr()
            m=LogisticRegression(C=C,max_iter=3000,class_weight='balanced').fit(A(Xtr,tr),yk)
            P.append(m.decision_function(A(Xva,va)))
        return np.array(CL)[np.argmax(P,0)]
    return fn
for n,f in [('union+vars C1',union(1)),('union+vars C0.3',union(0.3)),('union sin vars',union(1,False)),('nblr+vars C1',nbsvm(True,1)),('nblr+vars C0.3',nbsvm(True,0.3)),('nblr sin vars C1',nbsvm(False,1))]: cv(n,f)
