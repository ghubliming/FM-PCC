import re
LOGS={'0408 (T=1 requested, ran 0.5)':'/workspaces/FM-PCC/temp/0408/18_03_01_eval_dpcc_job_24254.log',
      '0508 (T=1 requested, ran 1.0)':'/workspaces/FM-PCC/temp/0508/00_36_44_eval_dpcc_job_24279.log'}
KEYS=[('Success rate:','SR'),('Constraints satisfied:','CS'),('Success rate (goal and constraints):','SR+C'),
      ('Avg number of steps:','steps'),('Avg number of constraint violations:','nviol'),
      ('Avg total violation:','totviol'),('Average computation time per step:','t/step')]
def parse(path):
    d={}; cur=None
    for line in open(path,errors='ignore'):
        m=re.match(r'-+Running (\S+) - (\S+) - (\S+) \((\d+)\)-+',line.strip())
        if m: cur=(m.group(2),m.group(3)); d[cur]={}; continue
        if cur is None: continue
        for k,short in KEYS:
            if line.startswith(k):
                d[cur][short]=line[len(k):].strip().split(' +-')[0].strip()
    return d
A=parse(LOGS['0408 (T=1 requested, ran 0.5)']); B=parse(LOGS['0508 (T=1 requested, ran 1.0)'])
order=[k for k in A]
for hs in ['top-right-hard','top-left-hard','both-hard']:
    print("="*112); print("HALFSPACE:",hs)
    hdr=f"{'variant':<26}"+"".join(f"{s:>9}/{s:<9}" for _,s in KEYS)
    print(f"{'variant':<26}"+"".join(f"{'old '+s:>12}{'new '+s:>12}" for _,s in KEYS))
    for (h,v) in order:
        if h!=hs: continue
        a=A[(h,v)]; b=B.get((h,v),{})
        row=f"{v:<26}"
        for _,s in KEYS:
            row+=f"{a.get(s,'-'):>12}{b.get(s,'-'):>12}"
        print(row)
