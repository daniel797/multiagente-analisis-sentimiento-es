import sys,numpy as np,pandas as pd,spacy,pickle
sys.path[:0]=['scripts/experimento']
import validacion_cruzada as VC, optimizar_hiperparametros as H
train,dev=H.cargar('data'); VC.Representacion.precalcular(train,dev)
nlp=spacy.load('es_core_news_md',disable=['ner','parser'])
out={}
for nom,df in [('train',train),('dev',dev)]:
    T=df.texto.map(lambda t:VC.V.aplicar(t,'base')).tolist()
    lem=[];vec=[]
    for d in nlp.pipe(T,batch_size=256):
        lem.append(' '.join(t.lemma_.lower() for t in d if not t.is_space))
        v=[t.vector for t in d if t.has_vector and not t.is_punct]
        vec.append(np.mean(v,0) if v else np.zeros(300))
    out[nom]=(lem,np.array(vec))
pickle.dump(out,open('/tmp/spacy_feats.pkl','wb'));print('ok')
