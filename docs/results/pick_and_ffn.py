import pandas as pd, numpy as np, warnings, sys; warnings.filterwarnings('ignore')
sys.path.insert(0,'/data3/akshatz/chemeleon_unlearn/src')
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from drift.descriptors import descriptor_modules
from drift.tasks import load_task
from drift import config as C
m=descriptor_modules(); cl=pd.read_parquet('results/corr_clusters.parquet').set_index('descriptor').cluster
sc=pd.read_csv('results/summary/phase2_relevance_scores.csv',index_col=0); n=sc.n; rk=sc.rank_consensus
D=pd.read_parquet(C.DATA/'descriptors'/'pubchem10k.parquet').iloc[:8000]
rep={}
for c,mem in cl.groupby(cl).groups.items():
    mem=list(mem); rep[c]=mem[0] if len(mem)==1 else D[mem].rank().corr().abs().mean().idxmax()
rep=pd.Series(rep)
def jac(a,b): return len(a&b)/len(a|b) if a|b else np.nan
print('=== Selection (1 representative per cluster) ===')
for k in [30,50,75,100,150]:
    TA=set(sc.auc_ames.nlargest(k).index); TL=set(sc.r2_ld50.nlargest(k).index); T=set(rk.nlargest(k).index)
    print(f"k={k:3d}: J(Ames,LD50)={jac(TA,TL):.2f} J(T,Ames)={jac(T,TA):.2f} J(T,LD50)={jac(T,TL):.2f} | CV-seed stability Ames {jac(TA,set(sc.auc_ames_s1.nlargest(k).index)):.2f} LD50 {jac(TL,set(sc.r2_ld50_s1.nlargest(k).index)):.2f} | covers {int(n.reindex(T).sum())} original descr")
g=pd.read_csv('results/summary/phase2_cluster_scores.csv',index_col=0)
sets={}
for k in [50,100]:
    T=list(rk.nlargest(k).index); G=list(g.gen.nlargest(k).index); R=list(np.random.default_rng(0).choice(rep.index,k,replace=False))
    sets[f'T{k}']=[rep[c] for c in T]; sets[f'G{k}']=[rep[c] for c in G]; sets[f'R{k}']=[rep[c] for c in R]
    print(f"k={k}: J(T,G)={jac(set(T),set(G)):.2f}")
sets['D535']=list(rep.values); sets['B1613']=list(cl.index)
tab=pd.DataFrame({'rep':rep.reindex(rk.nlargest(100).index),'cluster_size':n.reindex(rk.nlargest(100).index).astype(int),'auc_ames':sc.auc_ames.reindex(rk.nlargest(100).index).round(3),'r2_ld50':sc.r2_ld50.reindex(rk.nlargest(100).index).round(3)})
tab['module']=[m[r] for r in tab.rep]; tab.to_csv('results/summary/phase2_T100.csv')
print('\nT100 module composition:', tab.module.value_counts().to_dict())
print('\nT top 40 (representative, cluster size, single-cluster Ames AUC / LD50 R²):'); print(tab.head(40).to_string())
print('\n=== Shallow FFN (fastprop-style, 1 hidden layer 256, 5-fold CV on Polaris train split) ===')
def prep(task):
    t=load_task(task); X=pd.read_parquet(C.DATA/'descriptors'/(C.slug(task)+'.parquet'))[cl.index.tolist()]
    y=t.y_raw; ok=~np.isnan(y); X=X[ok]; y=y[ok]
    X=X.replace([np.inf,-np.inf],np.nan); mu=X.mean(); sd=X.std().replace(0,1); return ((X-mu)/sd).clip(-6,6).fillna(0), y
res=[]
for task,kind in [('tdcommons/ames','cls'),('tdcommons/ld50-zhu','reg')]:
    X,y=prep(task)
    for name,cols in sets.items():
        vals=[]
        for seed in (0,1,2):
            cv=StratifiedKFold(5,shuffle=True,random_state=seed) if kind=='cls' else KFold(5,shuffle=True,random_state=seed)
            if kind=='cls': mdl=MLPClassifier(hidden_layer_sizes=(256,),alpha=1e-3,max_iter=300,early_stopping=True,random_state=seed); s=cross_val_score(mdl,X[cols].values,(y>0.5).astype(int),cv=cv,scoring='roc_auc').mean()
            else: mdl=MLPRegressor(hidden_layer_sizes=(256,),alpha=1e-3,max_iter=300,early_stopping=True,random_state=seed); s=cross_val_score(mdl,X[cols].values,y,cv=cv,scoring='r2').mean()
            vals.append(s)
        res.append(dict(task=task.split('/')[1],set=name,n_descr=len(cols),score=np.mean(vals),sd=np.std(vals)))
        print(f"  {task.split('/')[1]:9s} {name:6s} n={len(cols):4d}  {'AUC' if kind=='cls' else 'R2'}={np.mean(vals):.3f} ± {np.std(vals):.3f}", flush=True)
pd.DataFrame(res).to_csv('results/summary/phase2_ffn_check.csv',index=False)
import json; json.dump({k:v for k,v in sets.items()}, open('results/summary/phase2_descriptor_sets.json','w'), indent=1)
