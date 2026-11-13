exec(open('/tmp/explora2.py').read().split("for n,f in [('union+vars")[0])
import pickle
S=pickle.load(open('/tmp/spacy_feats.pkl','rb')); LEM=pd.Series(S['train'][0]); EMB=S['train'][1]
def gen(use_lem=False,char=True,emb=False,vars_=True,C=1,wordtf=False,embw=1.0):
    def fn(tr,va):
        parts=[];
        txt=LEM if use_lem else T
        txt=txt.reset_index(drop=True)
        if wordtf or not char:
            vw=VC._vec_palabras().fit(txt.iloc[tr]); parts.append(lambda i,vw=vw:vw.transform(txt.iloc[i]))
        if char:
            vc=VC._vec_caracteres().fit(T.reset_index(drop=True).iloc[tr]); parts.append(lambda i,vc=vc:vc.transform(T.reset_index(drop=True).iloc[i]))
        if vars_:
            N=num(tr,va); parts.append(N)
        if emb:
            sc=StandardScaler().fit(EMB[tr]); parts.append(lambda i,sc=sc:csr_matrix(sc.transform(EMB[i])*embw))
        X=lambda i:hstack([p(i) for p in parts]).tocsr()
        return VC._lr(C).fit(X(tr),y[tr]).predict(X(va))
    return fn
for n,f in [('lem palabras+vars',gen(True,False,False,True)),('lem palabras+char+vars',gen(True,True,False,True,wordtf=True)),
 ('char+emb',gen(False,True,True,False)),('char+vars+emb C1',gen(False,True,True,True,1,embw=0.3)),('char+vars+emb C.3',gen(False,True,True,True,0.3,embw=0.3)),
 ('emb solo',gen(False,False,True,False,wordtf=False)),('lem+char+vars+emb C.3',gen(True,True,True,True,0.3,wordtf=True,embw=0.3))]:
    try: cv(n,f)
    except Exception as e: print(n,'ERR',e)
