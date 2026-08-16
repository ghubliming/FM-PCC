import zipfile, hashlib
BASE="/workspaces/FM-PCC/temp"
SKIP={"avg_time.npy"}
V=["dpcc-r","dpcc-r-tightened","dpcc-c","dpcc-c-tightened","dpcc-t","dpcc-t-tightened",
   "diffuser","gradient","gradient-tightened","post_processing","post_processing-tightened",
   "model_free","model_free-tightened"]
HS=["top-right-hard","top-left-hard","both-hard"]
def p(run,hs,v): return f"{BASE}/{run}/H8_K20_T1_Dmodels.GaussianDiffusion/6/results/halfspace_{hs}/{v}.npz"
def mem(path):
    z=zipfile.ZipFile(path)
    return {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n not in SKIP}
def sig(path):
    m=mem(path); h=hashlib.sha256()
    for k in sorted(m): h.update(k.encode()+m[k].encode())
    return h.hexdigest()[:12]

print("### A. BROKEN(0408 T1) vs FIXED(0508 T1) -- content only, avg_time excluded")
print(f"{'variant':<28}{'0408':<15}{'0508':<15}verdict   changed-members")
for hs in HS:
    print(f"-- {hs}")
    for v in V:
        a,b=mem(p('0408',hs,v)),mem(p('0508',hs,v))
        ch=sorted(k for k in a if a[k]!=b.get(k))
        sa=sig(p('0408',hs,v)); sb=sig(p('0508',hs,v))
        print(f"{v:<28}{sa:<15}{sb:<15}{'SAME' if sa==sb else 'CHANGED':<10}{','.join(x.replace('.npy','') for x in ch) or '-'}")

print()
print("### B. WITHIN-RUN identity: is post_processing a duplicate of dpcc-r or of diffuser?")
print(f"{'run':<8}{'halfspace':<18}{'pp==dpcc-r':<14}{'pp==diffuser':<14}{'pp-t==dpcc-r-t':<16}")
for run in ['0408','0508']:
    for hs in HS:
        pp=sig(p(run,hs,'post_processing')); dr=sig(p(run,hs,'dpcc-r')); df=sig(p(run,hs,'diffuser'))
        ppt=sig(p(run,hs,'post_processing-tightened')); drt=sig(p(run,hs,'dpcc-r-tightened'))
        print(f"{run:<8}{hs:<18}{str(pp==dr):<14}{str(pp==df):<14}{str(ppt==drt):<16}")
