import json,collections,csv,math,os
_HERE=os.path.dirname(os.path.abspath(__file__))
rows=json.load(open(os.path.join(_HERE,'gen_DA_20260818_rows.json')))
SCENES=sorted({r['scene'] for r in rows})
IDX={(r['cad'],r['K'],r['var'],r['scene']):r for r in rows}
_ag=collections.defaultdict(list)
for r in rows: _ag[(r['cad'],r['K'],r['var'])].append(r)
def M(c,K,v,k):
    rs=_ag.get((c,K,v))
    if not rs: return None
    x=[a[k] for a in rs if a[k] is not None]
    return sum(x)/len(x) if x else None
VARS=sorted({r['var'] for r in rows})

_h8=collections.defaultdict(lambda: collections.defaultdict(list))
CF='/workspaces/FM-PCC/temp/1708/batch_avoiding_combined_20260817_092728/candidates_multidimensional_aggregated.csv'
for r in csv.DictReader(open(CF)):
    for K in (1,2,5):
        if r['Folder_Name']==f'H8_K{K}_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE_msg20trials':
            v=r['variant']+('-tightened' if r['constraint_type']=='tightened' else '')
            _h8[(K,v)][r['metric']].append(float(r['mean']))
_KEY={'sc':'n_success_and_constraints','steps':'n_steps','t_step':'avg_time','nviol':'n_violations'}
def M8(K,v,k):
    d=_h8.get((K,v)); m=_KEY[k]
    if not d or m not in d: return None
    return sum(d[m])/len(d[m])

def plans_per_ep(cad,K,var):
    n=1 if cad=='r1' else 8
    return sum(math.ceil(IDX[(cad,K,var,s)]['steps']/n) for s in SCENES)/len(SCENES)
def per_plan(cad,K,var,fld):
    n=1 if cad=='r1' else 8
    tot=sum(IDX[(cad,K,var,s)][fld] for s in SCENES)
    pl=sum(2*math.ceil(IDX[(cad,K,var,s)]['steps']/n) for s in SCENES)   # n_trials=2
    return tot/pl
def fails_per_ep(cad,K,var):
    return sum(IDX[(cad,K,var,s)]['nlpf'] for s in SCENES)/6.0

def _betacdf(k,p,n):
    return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(0,k+1))
def clopper(k,n,a=0.05):
    lo,hi=0.0,1.0
    if k>0:
        f=lambda p:_betacdf(k-1,p,n)-(1-a/2)
        x,y=0.0,1.0
        for _ in range(200):
            m=(x+y)/2
            if f(m)>0: x=m
            else: y=m
        lo=(x+y)/2
    if k<n:
        f=lambda p:_betacdf(k,p,n)-(a/2)
        x,y=0.0,1.0
        for _ in range(200):
            m=(x+y)/2
            if f(m)>0: x=m
            else: y=m
        hi=(x+y)/2
    return lo,hi

NA={1:1,2:1,5:3}
def nproj(K): return K-int((1-0.5)*K)
def decomp(cfg,K):
    """returns u, c_slsqp, c_ipopt using the '-c' arms."""
    if cfg=='H8':  dif,dp,hf=M8(K,'diffuser','t_step'),M8(K,'dpcc-c','t_step'),M8(K,'hardflow_new-c','t_step')
    else:          dif,dp,hf=M('r1',K,'diffuser','t_step'),M('r1',K,'dpcc-c','t_step'),M('r1',K,'hardflow_new-c','t_step')
    na=NA[K]; u=dif/K
    return u,(dp-K*u)/(na*4),(hf-(K+na)*u)/na,dp,hf

TRACK={(1,'r1'):[0.041]*3,(1,'r8'):[0.050]*3,(2,'r1'):[0.040]*3,(2,'r8'):[0.040,0.062,0.062],
       (5,'r1'):[0.030]*3,(5,'r8'):[0.169,0.055,0.169]}

def best_tight(getter,fam,K):
    c=[(getter(K,v,'sc'),-getter(K,v,'steps'),-getter(K,v,'t_step'),v)
       for v in VARS if v.startswith(fam) and v.endswith('-tightened') and getter(K,v,'sc') is not None]
    b=max(c); return b[3],b[0],-b[1],-b[2]
def cheapest_tight(getter,fam,K):
    c=[(getter(K,v,'t_step'),v) for v in VARS if v.startswith(fam) and v.endswith('-tightened') and getter(K,v,'t_step') is not None]
    b=min(c); return b[1],getter(K,b[1],'sc'),getter(K,b[1],'steps'),b[0]
G8   = lambda K,v,k: M8(K,v,k)
G16r1= lambda K,v,k: M('r1',K,v,k)
G16r8= lambda K,v,k: M('r8',K,v,k)
