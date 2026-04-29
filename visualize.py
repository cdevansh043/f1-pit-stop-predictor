import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f', 'axes.facecolor': '#1a1a1a',
    'axes.edgecolor': '#444', 'axes.labelcolor': 'white',
    'xtick.color': 'white', 'ytick.color': 'white',
    'text.color': 'white', 'grid.color': '#333',
    'grid.linestyle': '--', 'font.family': 'monospace'
})
F1_RED = '#e8002d'
F1_SILVER = '#c0c0c0'
TEAL = '#00d4aa'
AMBER = '#f5a623'

_, X_test, _, y_test, FEATURES = joblib.load('processed_data.pkl')
results = joblib.load('model_results.pkl')
df = pd.read_csv('f1_features.csv').dropna(subset=FEATURES)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Model Comparison Dashboard
# ══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(18, 12))
fig1.suptitle('F1 Pit Stop Predictor — Model Comparison Dashboard',
              fontsize=18, fontweight='bold', color=F1_RED, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.35)

model_names = list(results.keys())
aucs        = [results[m]['auc'] for m in model_names]
f1_scores   = [results[m]['report']['1']['f1-score'] for m in model_names]
precisions  = [results[m]['report']['1']['precision'] for m in model_names]
recalls     = [results[m]['report']['1']['recall'] for m in model_names]
colors      = [F1_RED, TEAL, AMBER]

# — AUC bar chart —
ax1 = fig1.add_subplot(gs[0, 0])
bars = ax1.bar(model_names, aucs, color=colors, edgecolor='white', linewidth=0.5)
ax1.set_ylim(0.5, 1.0)
ax1.set_title('ROC-AUC Score', fontweight='bold', color=F1_SILVER)
ax1.set_ylabel('AUC')
for bar, val in zip(bars, aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, color='white')
ax1.tick_params(axis='x', rotation=15)

# — F1 / Precision / Recall grouped bar —
ax2 = fig1.add_subplot(gs[0, 1])
x = np.arange(len(model_names))
w = 0.25
ax2.bar(x - w, f1_scores,  w, label='F1',        color=TEAL,   edgecolor='white', linewidth=0.4)
ax2.bar(x,     precisions, w, label='Precision',  color=AMBER,  edgecolor='white', linewidth=0.4)
ax2.bar(x + w, recalls,    w, label='Recall',     color=F1_RED, edgecolor='white', linewidth=0.4)
ax2.set_xticks(x); ax2.set_xticklabels(model_names, rotation=15)
ax2.set_ylim(0, 1.05)
ax2.set_title('Pit-Lap Class Metrics', fontweight='bold', color=F1_SILVER)
ax2.legend(fontsize=8)

# — ROC curves all models —
ax3 = fig1.add_subplot(gs[0, 2])
for (name, res), col in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax3.plot(fpr, tpr, color=col, lw=2, label=f"{name} ({res['auc']:.3f})")
ax3.plot([0,1],[0,1],'--', color='#666', lw=1)
ax3.set_xlabel('False Positive Rate'); ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves', fontweight='bold', color=F1_SILVER)
ax3.legend(fontsize=8)

# — Confusion matrices (bottom row, one per model) —
for idx, (name, res) in enumerate(results.items()):
    ax = fig1.add_subplot(gs[1, idx])
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                linewidths=0.5, linecolor='#333',
                xticklabels=['No Pit','Pit'], yticklabels=['No Pit','Pit'],
                cbar=False)
    ax.set_title(f'Confusion Matrix\n{name}', fontweight='bold', color=F1_SILVER)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.savefig('fig1_model_comparison.png', dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
plt.show()
print("Saved fig1_model_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — XGBoost Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
xgb_model   = results['XGBoost']['model']
importances = xgb_model.feature_importances_
feat_df     = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
feat_df     = feat_df.sort_values('Importance', ascending=True)

fig2, ax = plt.subplots(figsize=(12, 8))
fig2.patch.set_facecolor('#0f0f0f')
bars = ax.barh(feat_df['Feature'], feat_df['Importance'],
               color=[F1_RED if v > feat_df['Importance'].median() else TEAL for v in feat_df['Importance']],
               edgecolor='white', linewidth=0.4)
for bar, val in zip(bars, feat_df['Importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8, color='white')
ax.set_xlabel('Feature Importance (XGBoost gain)')
ax.set_title('Feature Importance — What Drives Pit Stop Predictions?',
             fontsize=14, fontweight='bold', color=F1_RED, pad=15)
ax.axvline(feat_df['Importance'].median(), color=AMBER, linestyle='--', lw=1.5, label='Median')
ax.legend()
plt.tight_layout()
plt.savefig('fig2_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.show()
print("Saved fig2_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Lap Timeline: Predicted vs Actual (VER, Bahrain 2023)
# ══════════════════════════════════════════════════════════════════════════════
race_df = df[(df['Driver'] == 'VER') & (df['RaceName'] == 'Bahrain') & (df['Year'] == 2023)].copy()
scaler  = joblib.load('scaler.pkl')
X_race  = scaler.transform(race_df[FEATURES])
xgb     = results['XGBoost']['model']
race_df['predicted_prob'] = xgb.predict_proba(X_race)[:, 1]
race_df['predicted_pit']  = xgb.predict(X_race)

fig3, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig3.patch.set_facecolor('#0f0f0f')
fig3.suptitle('VER — 2023 Bahrain Grand Prix: Pit Stop Prediction Timeline',
              fontsize=15, fontweight='bold', color=F1_RED, y=1.01)

laps = race_df['LapNumber'].values

# Panel 1 — Lap time + pit markers
axes[0].plot(laps, race_df['lap_time_seconds'], color=TEAL, lw=1.5, label='Lap Time (s)')
axes[0].plot(laps, race_df['rolling_avg_lap_time'], color=AMBER, lw=1, linestyle='--', label='Rolling Avg')
for lap in race_df[race_df['is_pit_lap'] == 1]['LapNumber']:
    axes[0].axvline(lap, color=F1_RED, lw=2, alpha=0.8)
axes[0].set_ylabel('Lap Time (s)'); axes[0].legend(fontsize=8)
axes[0].set_title('Lap Time vs Rolling Average  |  Red = Actual Pit Lap', color=F1_SILVER)

# Panel 2 — Pit probability
axes[1].fill_between(laps, race_df['predicted_prob'], alpha=0.4, color=F1_RED)
axes[1].plot(laps, race_df['predicted_prob'], color=F1_RED, lw=1.5, label='Pit Probability')
axes[1].axhline(0.5, color=AMBER, linestyle='--', lw=1, label='Decision threshold (0.5)')
for lap in race_df[race_df['is_pit_lap'] == 1]['LapNumber']:
    axes[1].axvline(lap, color=F1_RED, lw=2, alpha=0.8)
axes[1].set_ylim(0, 1); axes[1].set_ylabel('Predicted Pit Prob')
axes[1].legend(fontsize=8)
axes[1].set_title('Model Pit Stop Probability per Lap', color=F1_SILVER)

# Panel 3 — Tyre life coloured by compound
compound_colours = {'SOFT': F1_RED, 'MEDIUM': AMBER, 'HARD': '#cccccc', 'INTERMEDIATE': '#00cc44', 'WET': '#4488ff'}
for _, row in race_df.iterrows():
    c = compound_colours.get(row['Compound'], 'white')
    axes[2].bar(row['LapNumber'], row['TyreLife'], color=c, width=0.9, alpha=0.85)
for lap in race_df[race_df['is_pit_lap'] == 1]['LapNumber']:
    axes[2].axvline(lap, color='white', lw=1.5, alpha=0.6)
axes[2].set_xlabel('Lap Number'); axes[2].set_ylabel('Tyre Life (laps)')
axes[2].set_title('Tyre Life per Lap  |  Colour = Compound', color=F1_SILVER)

# Compound legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k, v in compound_colours.items()]
axes[2].legend(handles=legend_elements, fontsize=8, loc='upper left')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig3_lap_timeline.png', dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
plt.show()
print("Saved fig3_lap_timeline.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Pit Stop Probability Distribution by Tyre Age & Race Progress
# ══════════════════════════════════════════════════════════════════════════════
xgb_proba = results['XGBoost']['y_proba']
plot_df   = pd.DataFrame({'prob': xgb_proba, 'actual': y_test.values}).reset_index(drop=True)
feat_test = pd.DataFrame(X_test, columns=FEATURES)
plot_df['tyre_life']     = feat_test['TyreLife'].values
plot_df['race_progress'] = feat_test['race_progress'].values

fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
fig4.patch.set_facecolor('#0f0f0f')
fig4.suptitle('Pit Stop Probability Distribution — What the Model Learned',
              fontsize=14, fontweight='bold', color=F1_RED)

# Left — by tyre life bucket
plot_df['tyre_bucket'] = pd.cut(plot_df['tyre_life'], bins=[0,5,10,15,20,25,60],
                                 labels=['1-5','6-10','11-15','16-20','21-25','25+'])
means = plot_df.groupby('tyre_bucket', observed=True)['prob'].mean()
axes[0].bar(means.index.astype(str), means.values, color=F1_RED, edgecolor='white', linewidth=0.5)
axes[0].set_xlabel('Tyre Age (laps)'); axes[0].set_ylabel('Avg Predicted Pit Probability')
axes[0].set_title('Pit Probability by Tyre Age', fontweight='bold', color=F1_SILVER)
for i, v in enumerate(means.values):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', color='white', fontsize=9)

# Right — by race progress bucket
plot_df['prog_bucket'] = pd.cut(plot_df['race_progress'], bins=10)
prog_means = plot_df.groupby('prog_bucket', observed=True)['prob'].mean()
labels = [f'{int(b.left*100)}-{int(b.right*100)}%' for b in prog_means.index]
axes[1].bar(labels, prog_means.values, color=TEAL, edgecolor='white', linewidth=0.5)
axes[1].set_xlabel('Race Progress'); axes[1].set_ylabel('Avg Predicted Pit Probability')
axes[1].set_title('Pit Probability by Race Progress', fontweight='bold', color=F1_SILVER)
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(prog_means.values):
    axes[1].text(i, v + 0.003, f'{v:.3f}', ha='center', color='white', fontsize=9)

for ax in axes:
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fig4_probability_distributions.png', dpi=150, bbox_inches='tight', facecolor=fig4.get_facecolor())
plt.show()
print("Saved fig4_probability_distributions.png")

print("\n✅ All 4 figures saved.")