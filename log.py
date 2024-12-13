# Extracting user responses for psychopy log data
import pandas as pd
processed_data_with_type = []
d = pd.read_csv('data.csv')
for idx, row in d.iterrows():
    if not pd.isna(row['current_hlt_stim']):
        user_value = row['trials_hlt.slider_4.response'] if not pd.isna(
            row['trials_hlt.slider_4.response']) else "No Value"
        user_rt = row['slider_4.rt'] if not pd.isna(
            row['slider_4.rt']) else "No RT"
        processed_data_with_type.append({
            "Stim Type": "HLT",
            "Stim Name": row['current_hlt_stim'],
            "Repeat Number": int(row['thisRepN']) + 1 if not pd.isna(row['thisRepN']) else None,
            "User Value": user_value,
            "User Response Time (s)": f"{user_rt:.3f}"
        })

    if not pd.isna(row['current_let_stim']):
        user_value = row['slider_4.response'] if not pd.isna(
            row['slider_4.response']) else "No Value"
        for response_col, rt_col in [('trials_let.slider_5.response', 'slider_5.rt'),
                                     ('trials_let.slider_6.response', 'slider_6.rt'),
                                     ('trials_let.slider_7.response', 'slider_7.rt')]:
            if not pd.isna(row[response_col]):
                user_value = row[response_col]
                user_rt = row[rt_col]
                break

        user_value = user_value if user_value is not None else "No Value"
        user_rt = user_rt if user_rt is not None else "No RT"
        processed_data_with_type.append({
            "Stim Type": "LET",
            "Stim Name": row['current_let_stim'],
            "Repeat Number": int(row['thisRepN']) + 1 if not pd.isna(row['thisRepN']) else None,
            "User Value": user_value,
            "User Response Time (s)": f"{user_rt:.3f}"
        })

processed_df_with_type = pd.DataFrame(processed_data_with_type)
processed_df_with_type.to_csv('processed_data_with_type.csv', index=False)
