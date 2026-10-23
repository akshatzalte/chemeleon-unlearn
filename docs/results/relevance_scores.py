import pandas as pd, numpy as np, warnings, sys; warnings.filterwarnings('ignore')
sys.path.insert(0,'/data3/akshatz/chemeleon_unlearn/src')
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from drift.descriptors import descriptor_modules
from drift.tasks import load_task
from drift import config as C
m=descriptor_modules(); cl=pd.read_parquet('results/corr_clusters.parquet').set_index('descriptor').cluster
def prep(task):
    t=load_task(task); D=pd.read_parquet(C.DATA/'descriptors'/(C.slug(task)+'.parquet'))[cl.index.tolist()]
    y=t.y_raw; ok=~np.isnan(y); D=D[ok]; y=y[ok]
    X=D.replace([np.inf,-np.inf],np.nan); mu=X.mean(); sd=X.std().replace(0,1); X=((X-mu)/sd).clip(-6,6).fillna(0)
    return X,y
def cluster_scores(X,y,kind,seed=0):
    out={}; cv=StratifiedKFold(5,shuffle=True,random_state=seed) if kind=='cls' else KFold(5,shuffle=True,random_state=seed)
    for c,mem in cl.groupby(cl).groups.items():
        Xi=X[list(mem)].values
        if kind=='cls': sc=cross_val_score(LogisticRegressionCV(Cs=5,cv=3,max_iter=500),Xi,(y>0.5).astype(int),cv=cv,scoring='roc_auc').mean()
        else: sc=cross_val_score(RidgeCV(alphas=np.logspace(-2,3,8)),Xi,y,cv=cv,scoring='r2').mean()
        out[c]=sc
    return pd.Series(out)
XA,yA=prep('tdcommons/ames'); XL,yL=prep('tdcommons/ld50-zhu')
sA=cluster_scores(XA,yA,'cls'); sL=cluster_scores(XL,yL,'reg'); sA2=cluster_scores(XA,yA,'cls',1); sL2=cluster_scores(XL,yL,'reg',1)
n=cl.value_counts(); rk=(sA.rank(pct=True)+sL.rank(pct=True))/2
pd.DataFrame({'auc_ames':sA,'r2_ld50':sL,'auc_ames_s1':sA2,'r2_ld50_s1':sL2,'rank_consensus':rk,'n':n.reindex(sA.index)}).to_csv('results/summary/phase2_relevance_scores.csv')
print('scores saved')
