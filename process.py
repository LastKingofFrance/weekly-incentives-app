import pandas as pd
from datetime import datetime
import io
import re
import numpy as np

TODAY = datetime.today()

REWARD_THRESHOLDS = {
    'junior': {'min': 42000, 'max': 72000, 'reward': 1000},
    'standard': [
        {'min': 360000, 'reward': 4000},
        {'min': 240000, 'reward': 3000},
        {'min': 180000, 'reward': 2000},
        {'min': 120000, 'reward': 1500},
        {'min': 72000, 'reward': 1000}
    ]
}

def clean_master(master_file):
    xl = pd.ExcelFile(master_file)
    df = xl.parse("Sheet2", skiprows=0)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Contract Date'])
    df['Contract Date'] = pd.to_datetime(df['Contract Date'], format='mixed', errors='coerce')
    df['Tenure'] = ((TODAY - df['Contract Date']).dt.days / 365).round(1)
    df['LevelCode'] = (
        df['LevelCode'].astype(str).str.strip()
        .str.extract(r'(\d+)')[0]
        .str.zfill(8)
    )
    df = df[df['LevelCode'].str.len() == 8]
    return df[['LevelCode', 'Agent Name', 'Agency', 'Tenure']]

def clean_nbt(nbt_file):
    xl = pd.ExcelFile(nbt_file)
    nbt_sheet = next((s for s in xl.sheet_names if re.match(r"^NBT\s*\d+[, ]\s*Q\d+$", s.strip(), re.IGNORECASE)), None)
    if not nbt_sheet:
        raise ValueError("❌ NBT sheet like 'NBT 9 Q2' not found.")
    df = xl.parse(nbt_sheet, dtype=str)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Policy No', 'LevelCode'])
    df['LevelCode'] = df['LevelCode'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str[-8:].str.zfill(8)
    df['EstimatedAPI'] = pd.to_numeric(df['EstimatedAPI'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    df['Count'] = pd.to_numeric(df['Count'].astype(str).str.replace(r'[^\d]', '', regex=True), errors='coerce')
    if (df['Count'] % 10 == 0).all() and (df['Count'] > 0).any():
        df['Count'] = df['Count'] / 10
    df = df.dropna(subset=['EstimatedAPI', 'Count'])
    return df[df['LevelCode'].str.len() == 8]

def extract_high_api_bonus(nbt_df, master_df):
    merged = nbt_df.merge(master_df, on='LevelCode', how='inner')
    high_api = merged[merged['EstimatedAPI'] >= 600000].copy()
    high_api['Bonus'] = 5000
    return high_api[['LevelCode', 'Agent Name', 'Agency', 'Tenure', 'Policy No', 'EstimatedAPI', 'Bonus']] \
        .sort_values(by='EstimatedAPI', ascending=False)

def calculate_agent_rewards(nbt_df, master_df):
    filtered_df = nbt_df[nbt_df['EstimatedAPI'] < 600000]
    merged = filtered_df.merge(master_df, on='LevelCode', how='inner')
    grouped = merged.groupby(['LevelCode', 'Agent Name', 'Agency', 'Tenure']) \
        .agg(EstimatedAPI=('EstimatedAPI', 'sum'), Lives=('Count', 'sum')).reset_index()
    conditions = [
        (grouped['Tenure'] < 3) & (grouped['EstimatedAPI'] >= 42000) & (grouped['EstimatedAPI'] < 72000),
        (grouped['EstimatedAPI'] >= 360000),
        (grouped['EstimatedAPI'] >= 240000),
        (grouped['EstimatedAPI'] >= 180000),
        (grouped['EstimatedAPI'] >= 120000),
        (grouped['EstimatedAPI'] >= 72000)
    ]
    rewards = [1000, 4000, 3000, 2000, 1500, 1000]
    grouped['Reward'] = np.select(conditions, rewards, default=0)
    return grouped.sort_values(by='Reward', ascending=False)

def get_agency_target_and_reward(agency_name):
    if not isinstance(agency_name, str):
        return (np.nan, np.nan)

    agency_upper = agency_name.upper().strip()

    if agency_upper == "MANAGERS2":
        return (np.nan, np.nan)

    if any(code in agency_upper for code in ["NAIROBI 1", "NAIROBI 2", "NAIROBI 5"]):
        return (2200000, 10000)
    elif any(code in agency_upper for code in ["NAIROBI 3", "NAIROBI 6", "NAIROBI 7"]):
        return (1600000, 8000)
    elif any(code in agency_upper for code in ["NAIROBI 4", "NAIROBI 9"]):
        return (1100000, 6000)
    elif "NAIROBI" not in agency_upper:
        return (550000, 4000)

    return (np.nan, np.nan)

def calculate_unit_leader_rewards(nbt_df, agency_unit_file, active_agents_file):
    xl = pd.ExcelFile(agency_unit_file)
    normalized_sheets = {s.strip().lower(): s for s in xl.sheet_names}
    if 'unit' not in normalized_sheets:
        raise ValueError("❌ Sheet 'Unit' not found.")
    unit_df = xl.parse(normalized_sheets['unit'], dtype=str)
    unit_df.columns = unit_df.columns.str.strip()
    unit_df = unit_df[['UL Code', 'Unit Code', 'Name']]
    unit_df.columns = ['ULCode', 'UnitCode', 'Leader']

    agents_df = pd.read_excel(active_agents_file, sheet_name=0, dtype=str)
    agents_df.columns = agents_df.columns.str.strip()
    agents_df = agents_df[['Agent Code', 'Unit Code', 'Agency']]
    agents_df.columns = ['LevelCode', 'UnitCode', 'Agency']
    agents_df['LevelCode'] = agents_df['LevelCode'].str.strip().str.zfill(8)
    merged = nbt_df.merge(agents_df, on='LevelCode', how='inner')
    unit_agg = merged.groupby(['UnitCode', 'Agency']).agg(AchievedAPI=('EstimatedAPI', 'sum')).reset_index()
    full_df = unit_agg.merge(unit_df, on='UnitCode', how='inner')
    full_df[['WeeklyTarget', 'BaseReward']] = full_df['Agency'].apply(lambda x: pd.Series(get_agency_target_and_reward(x)))
    full_df['Percentage'] = (full_df['AchievedAPI'] / full_df['WeeklyTarget']).round(2)
    full_df['Reward'] = full_df.apply(lambda row: row['BaseReward'] if row['Percentage'] >= 0.9 and row['AchievedAPI'] >= row['WeeklyTarget'] else 0, axis=1)
    return full_df[['UnitCode', 'Leader', 'Agency', 'WeeklyTarget', 'AchievedAPI', 'Percentage', 'Reward']] \
        .sort_values(by='Reward', ascending=False)

def calculate_agency_rewards(nbt_df, master_df):
    merged = nbt_df.merge(master_df[['LevelCode', 'Agency']], on='LevelCode', how='inner')
    agency_grouped = merged.groupby('Agency').agg(AchievedAPI=('EstimatedAPI', 'sum')).reset_index()
    agency_grouped[['WeeklyTarget', 'RewardBase']] = agency_grouped['Agency'].apply(lambda x: pd.Series(get_agency_target_and_reward(x)))
    agency_grouped['Percentage'] = (agency_grouped['AchievedAPI'] / agency_grouped['WeeklyTarget']).round(2)
    agency_grouped['Reward'] = agency_grouped.apply(
        lambda row: row['RewardBase'] if row['Percentage'] >= 0.9 and row['AchievedAPI'] >= row['WeeklyTarget'] else 0,
        axis=1
    )
    return agency_grouped[['Agency', 'WeeklyTarget', 'AchievedAPI', 'Percentage', 'Reward']] \
        .sort_values(by='Reward', ascending=False)

def format_excel_sheet(writer, sheet_name, df, percent_cols=None):
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    money_fmt = '#,##0'
    percent_fmt = '0%'

    for idx, col in enumerate(df.columns):
        col_letter = chr(65 + idx)
        if df[col].dtype in ['int64', 'float64'] and (percent_cols is None or col not in percent_cols):
            worksheet.column_dimensions[col_letter].width = 15
            for cell in worksheet[col_letter][1:]:
                cell.number_format = money_fmt
        elif percent_cols and col in percent_cols:
            worksheet.column_dimensions[col_letter].width = 12
            for cell in worksheet[col_letter][1:]:
                cell.number_format = percent_fmt

def process_files(nbt_file, agency_unit_file, active_agents_file, master_file):
    nbt_df = clean_nbt(nbt_file)
    master_df = clean_master(master_file)

    agent_reward_df = calculate_agent_rewards(nbt_df, master_df)
    high_api_df = extract_high_api_bonus(nbt_df, master_df)
    unit_leader_df = calculate_unit_leader_rewards(nbt_df, agency_unit_file, active_agents_file)
    agency_reward_df = calculate_agency_rewards(nbt_df, master_df)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        format_excel_sheet(writer, 'Agent Reward', agent_reward_df)
        format_excel_sheet(writer, 'High API Bonuses', high_api_df)
        format_excel_sheet(writer, 'Unit Leader Reward', unit_leader_df, percent_cols=['Percentage'])
        format_excel_sheet(writer, 'Agency Reward', agency_reward_df, percent_cols=['Percentage'])

    output.seek(0)
    return output
