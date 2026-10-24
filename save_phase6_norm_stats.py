import json
import pandas as pd

master_df = pd.read_csv('full_dataset_4407_folds.csv')
rad_tab_df = pd.read_csv('dicom_and_radiomics_features.csv')
df = pd.merge(master_df, rad_tab_df, on='StudyInstanceUID', how='left')

df['ratio_sag_total'] = df['n_sagittal_slices'] / (df['n_total_slices'] + 1e-5)
df['ratio_cor_total'] = df['n_coronal_slices'] / (df['n_total_slices'] + 1e-5)
df['ratio_ax_total'] = df['n_axial_slices'] / (df['n_total_slices'] + 1e-5)
df['tr_te_ratio'] = df['mean_tr'] / (df['mean_te'] + 1e-5)
df = pd.get_dummies(df, columns=['manufacturer'], drop_first=True)

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]
exclude = ['StudyInstanceUID', 'fold', 'is_gold', 'Report', 'has_report'] + target_cols
rad_cols = [c for c in df.columns if c not in exclude]

dicom_cols = [
    'n_series', 'n_total_slices', 'n_sagittal_slices', 'n_coronal_slices', 'n_axial_slices',
    'ratio_sag_total', 'ratio_cor_total', 'ratio_ax_total',
    'patient_sex', 'field_strength', 'mean_tr', 'mean_te', 'pixel_spacing', 'tr_te_ratio'
] + [c for c in df.columns if c.startswith('manufacturer_')]

stats = {
    'rad_cols': rad_cols,
    'rad_means': df[rad_cols].mean().to_dict(),
    'rad_stds': df[rad_cols].std().to_dict(),
    'dicom_cols': dicom_cols,
    'dicom_means': df[dicom_cols].mean().to_dict(),
    'dicom_stds': df[dicom_cols].std().to_dict(),
}

with open('checkpoints/phase6_norm_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print('Saved phase6_norm_stats.json! Features count: rad=', len(rad_cols), 'dicom=', len(dicom_cols))
