# ============================================================
# Project: Unemployment Analysis with Python
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────
df1 = pd.read_csv('Unemployment_in_India.csv')
df2 = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')

# Clean column names (strip spaces)
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

print("=" * 60)
print("       UNEMPLOYMENT ANALYSIS — INDIA (Covid-19 Impact)")
print("=" * 60)

print("\n[Dataset 1] Unemployment in India")
print(f"  Shape   : {df1.shape}")
print(f"  Columns : {list(df1.columns)}")
print(f"\n{df1.head(3)}")

print("\n[Dataset 2] Unemployment Rate upto Nov-2020")
print(f"  Shape   : {df2.shape}")
print(f"  Columns : {list(df2.columns)}")
print(f"\n{df2.head(3)}")

# ── 2. CLEAN & PREPARE ───────────────────────────────────────
# Dataset 2 has geo info → richer for time-series analysis
df2['Date'] = pd.to_datetime(df2['Date'].str.strip(), dayfirst=True)
df2 = df2.sort_values('Date')

unemp_col  = 'Estimated Unemployment Rate (%)'
employ_col = 'Estimated Employed'

print(f"\nDate range: {df2['Date'].min().date()} → {df2['Date'].max().date()}")
print(f"Regions   : {df2['Region'].nunique()}")
print(f"\nMissing values:\n{df2[[unemp_col, employ_col]].isnull().sum()}")

# Monthly national average
monthly = df2.groupby('Date')[unemp_col].mean().reset_index()
monthly.columns = ['Date', 'Avg_Unemployment_Rate']

# ── 3. SUMMARY STATS ─────────────────────────────────────────
pre_covid  = monthly[monthly['Date'] < '2020-03-01']['Avg_Unemployment_Rate']
post_covid = monthly[monthly['Date'] >= '2020-03-01']['Avg_Unemployment_Rate']

print("\n--- Summary Statistics ---")
print(f"Pre-Covid  avg unemployment  : {pre_covid.mean():.2f}%")
print(f"Post-Covid avg unemployment  : {post_covid.mean():.2f}%")
print(f"Peak unemployment rate       : {monthly['Avg_Unemployment_Rate'].max():.2f}%  "
      f"({monthly.loc[monthly['Avg_Unemployment_Rate'].idxmax(), 'Date'].strftime('%b %Y')})")

# ── 4. VISUALISATIONS ────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle('India Unemployment Analysis — Covid-19 Impact',
             fontsize=17, fontweight='bold', y=0.98)

# ── Plot 1: Monthly trend with Covid marker
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(monthly['Date'], monthly['Avg_Unemployment_Rate'],
         color='#2c3e50', linewidth=2, marker='o', markersize=4)
ax1.axvline(pd.Timestamp('2020-03-01'), color='red',
            linestyle='--', linewidth=1.5, label='Covid Lockdown (Mar 2020)')
ax1.fill_between(monthly['Date'], monthly['Avg_Unemployment_Rate'],
                 alpha=0.15, color='#3498db')
ax1.set_title('Monthly Avg Unemployment Rate')
ax1.set_xlabel('Date')
ax1.set_ylabel('Unemployment Rate (%)')
ax1.legend(fontsize=8)
ax1.tick_params(axis='x', rotation=30)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

# ── Plot 2: Top 10 regions — highest avg unemployment
ax2 = fig.add_subplot(2, 2, 2)
region_avg = (df2.groupby('Region')[unemp_col]
              .mean()
              .sort_values(ascending=False)
              .head(10))
colors_bar = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(region_avg)))
region_avg.plot(kind='bar', ax=ax2, color=colors_bar, edgecolor='white')
ax2.set_title('Top 10 Regions by Avg Unemployment Rate')
ax2.set_xlabel('Region')
ax2.set_ylabel('Avg Unemployment Rate (%)')
ax2.tick_params(axis='x', rotation=40)
for bar in ax2.patches:
    ax2.annotate(f'{bar.get_height():.1f}%',
                 (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 ha='center', va='bottom', fontsize=7)

# ── Plot 3: Pre vs Post Covid comparison box
ax3 = fig.add_subplot(2, 2, 3)
df2['Period'] = df2['Date'].apply(
    lambda x: 'Post-Covid\n(Mar 2020+)' if x >= pd.Timestamp('2020-03-01')
    else 'Pre-Covid\n(Before Mar 2020)')
period_data = [
    df2[df2['Period'].str.startswith('Pre')][unemp_col].dropna(),
    df2[df2['Period'].str.startswith('Post')][unemp_col].dropna()
]
bp = ax3.boxplot(period_data, labels=['Pre-Covid', 'Post-Covid'],
                 patch_artist=True, notch=False)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
ax3.set_title('Unemployment Distribution: Pre vs Post Covid')
ax3.set_ylabel('Unemployment Rate (%)')

# ── Plot 4: Area-wise (Rural / Urban if in df1)
ax4 = fig.add_subplot(2, 2, 4)
df1['Date'] = pd.to_datetime(df1['Date'].str.strip(), dayfirst=True)
unemp_col1 = 'Estimated Unemployment Rate (%)'
if 'Area' in df1.columns:
    area_monthly = (df1.groupby(['Date', 'Area'])[unemp_col1]
                    .mean().reset_index())
    for area, color in zip(['Rural', 'Urban'], ['#27ae60', '#8e44ad']):
        sub = area_monthly[area_monthly['Area'] == area]
        if not sub.empty:
            ax4.plot(sub['Date'], sub[unemp_col1],
                     label=area, color=color, linewidth=2)
    ax4.set_title('Rural vs Urban Unemployment Trend')
    ax4.set_ylabel('Unemployment Rate (%)')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=30)
else:
    ax4.text(0.5, 0.5, 'Area column not available', transform=ax4.transAxes,
             ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig('unemployment_analysis_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[Saved] unemployment_analysis_results.png")

# ── 5. KEY INSIGHTS ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  KEY INSIGHTS")
print("=" * 60)
print(f"1. Pre-Covid average unemployment  : {pre_covid.mean():.2f}%")
print(f"2. Post-Covid average unemployment : {post_covid.mean():.2f}%")
increase = post_covid.mean() - pre_covid.mean()
print(f"3. Increase due to Covid           : +{increase:.2f} percentage points")
peak_month = monthly.loc[monthly['Avg_Unemployment_Rate'].idxmax(), 'Date']
print(f"4. Highest spike month             : {peak_month.strftime('%B %Y')}")
print(f"5. Most affected region            : {region_avg.idxmax()}")
print("\nProject Complete!")
