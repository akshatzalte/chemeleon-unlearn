import pandas as pd, numpy as np, warnings, sys; warnings.filterwarnings('ignore')
sys.path.insert(0,'/data3/akshatz/chemeleon_unlearn/src')
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from drift.tasks import load_task
from drift import config as C
import json
cl=pd.read_parquet('results/corr_clusters.parquet').set_index('descriptor').cluster
sc=pd.read_csv('results/summary/phase2_relevance_scores.csv',index_col=0)
sets=json.load(open('results/summary/phase2_descriptor_sets.json')); rep_all=sets['D535']
rep=pd.Series({cl[r]:r for r in rep_all})
def prep(task):
    t=load_task(task); X=pd.read_parquet(C.DATA/'descriptors'/(C.slug(task)+'.parquet'))[cl.index.tolist()]
    y=t.y_raw; ok=~np.isnan(y); X=X[ok]; y=y[ok]
    X=X.replace([np.inf,-np.inf],np.nan); mu=X.mean(); sd=X.std().replace(0,1); return ((X-mu)/sd).clip(-6,6).fillna(0), y
ks=[10,20,30,50,75,100,150,200,300,535]
rows=[]
for task,kind,rank in [('tdcommons/ames','cls',sc.auc_ames),('tdcommons/ld50-zhu','reg',sc.r2_ld50)]:
    X,y=prep(task); yy=(y>0.5).astype(int) if kind=='cls' else y
    order=rank.sort_values(ascending=False).index
    for k in ks:
        cols=[rep[c] for c in order[:k]]
        rcols=[rep[c] for c in np.random.default_rng(0).choice(rep.index,k,replace=False)]
        for name,cc in [('T',cols),('R',rcols)]:
            lin=[];ffn=[]
            for seed in (0,1,2):
                cv=StratifiedKFold(5,shuffle=True,random_state=seed) if kind=='cls' else KFold(5,shuffle=True,random_state=seed)
                if kind=='cls':
                    lin.append(cross_val_score(LogisticRegressionCV(Cs=5,cv=3,max_iter=1000),X[cc].values,yy,cv=cv,scoring='roc_auc').mean())
                    if k in (20,50,100,200,535) and name=='T': ffn.append(cross_val_score(MLPClassifier(hidden_layer_sizes=(256,),alpha=1e-3,max_iter=300,early_stopping=True,random_state=seed),X[cc].values,yy,cv=cv,scoring='roc_auc').mean())
                else:
                    lin.append(cross_val_score(RidgeCV(alphas=np.logspace(-2,3,8)),X[cc].values,yy,cv=cv,scoring='r2').mean())
                    if k in (20,50,100,200,535) and name=='T': ffn.append(cross_val_score(MLPRegressor(hidden_layer_sizes=(256,),alpha=1e-3,max_iter=300,early_stopping=True,random_state=seed),X[cc].values,yy,cv=cv,scoring='r2').mean())
            rows.append(dict(task=task.split('/')[1],set=name,k=k,lin=np.mean(lin),lin_sd=np.std(lin),ffn=np.mean(ffn) if ffn else np.nan))
            print(f"{task.split('/')[1]:9s} {name} k={k:3d} linear={np.mean(lin):.3f}±{np.std(lin):.3f}" + (f" ffn={np.mean(ffn):.3f}" if ffn else ""), flush=True)
df=pd.DataFrame(rows); df.to_csv('results/summary/phase2_k_sweep.csv',index=False)
for task in df.task.unique():
    t=df[(df.task==task)&(df.set=='T')]; best=t.lin.max(); tol=t.lin_sd.max()
    kstar=int(t[t.lin>=best-tol].k.min()); print(f"\n{task}: linear best {best:.3f} at k={int(t.loc[t.lin.idxmax(),'k'])}; smallest k within 1 SD of best: k*={kstar}")
