# Analyze upsets in Rank regime
import pandas as pd

df = pd.read_csv('d:/2026-repo/data/2026_MCM_Problem_C_Data_Cleaned.csv')

print("=" * 60)
print("Rank 制度 (S1-2) 淘汰分析: 哪些是'爆冷'?")
print("=" * 60)

upsets = []
expected = []

for s in [1, 2]:
    print(f"\n=== Season {s} ===")
    sdf = df[df['season'] == s].copy()
    weeks = sorted(set([int(c[1:]) for c in sdf.columns if c.startswith('J') and c[1:].isdigit()]))
    
    for w in weeks:
        col = f'J{w}'
        active = sdf[sdf[col] > 0].copy()
        n = len(active)
        if n == 0:
            continue
        
        J = active[col].values
        names = active['celebrity_name'].tolist()
        
        # Judge rank (1=best)
        j_order = list(reversed(sorted(range(n), key=lambda i: J[i])))
        j_ranks = {j_order[r]: r+1 for r in range(n)}
        
        elim = active[(active['elim_week'] == w) & (~active['withdrew'])]
        if len(elim) == 0:
            continue
        
        elim_name = elim['celebrity_name'].iloc[0]
        elim_idx = names.index(elim_name)
        elim_jrank = j_ranks[elim_idx]
        
        # 如果淘汰者不是 judge rank 最差的，就是"爆冷"
        worst_jrank = n
        is_upset = (elim_jrank < worst_jrank)
        
        status = "UPSET" if is_upset else "expected"
        print(f"  Week {w}: {elim_name:15s} j_rank={elim_jrank}/{n} -> {status}")
        
        if is_upset:
            upsets.append((s, w, elim_name, elim_jrank, n))
        else:
            expected.append((s, w, elim_name))

print("\n" + "=" * 60)
print(f"Summary: {len(expected)} expected, {len(upsets)} upsets")
print("=" * 60)
print("\nUpset cases (fan vote overcame judge score):")
for s, w, name, jr, n in upsets:
    print(f"  S{s}W{w}: {name} had j_rank={jr}/{n}, still eliminated")
    print(f"         -> Strong evidence of low fan support!")
