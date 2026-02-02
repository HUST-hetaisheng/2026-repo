import matplotlib.pyplot as plt

# Re-filter for target seasons mentioned in the prompt
# Jerry Rice: S2, Billy Ray Cyrus: S4, Bristol Palin: S11, Bobby Bones: S27, Nelly: S29, Iman Shumpert: S30
specific_targets = [
    ("Jerry Rice", 2),
    ("Billy Ray Cyrus", 4),
    ("Bristol Palin", 11),
    ("Bobby Bones", 27),
    ("Nelly", 29),
    ("Iman Shumpert", 30)
]

plt.figure(figsize=(12, 8))

for i, (name, season) in enumerate(specific_targets):
    data = final_view[(final_view['celebrity_name'] == name) & (final_view['season'] == season)].sort_values('week')
    
    plt.subplot(3, 2, i+1)
    plt.plot(data['week'], data['cum_p_rank'], 'o-', label='Rank Rule', color='blue', alpha=0.7)
    plt.plot(data['week'], data['cum_p_pct'], 's--', label='Percent Rule', color='orange', alpha=0.7)
    
    plt.title(f"{name} (Season {season})")
    plt.yscale('log') # Log scale to see the decay clearly
    plt.xlabel('Week')
    plt.ylabel('Cum. Survival P')
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig('probabilistic_survival_comparison.png')