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

def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")
    return df

def clean_master(master_file):
    xl = pd.ExcelFile(master_file)
    df = xl.parse("Sheet2", skiprows=0)
    df = standardize_columns(df)
    if 'contractdate' not in df.columns or 'levelcode' not in df.columns:
        raise ValueError("Missing expected columns in master file")
    df = df.dropna(subset=['contractdate'])
    df['contractdate'] = pd.to_datetime(df['contractdate'], format='mixed', errors='coerce')
    df['tenure'] = ((TODAY - df['contractdate']).dt.days / 365).round(1)
    df['levelcode'] = (
        df['levelcode'].astype(str).str.strip()
        .str.extract(r'(\d+)')[0]
        .str.zfill(8)
    )
    df = df[df['levelcode'].str.len() == 8]
    return df[['levelcode', 'agentname', 'agency', 'tenure']]

def clean_nbt(nbt_file):
    xl = pd.ExcelFile(nbt_file)
    nbt_sheet = next((s for s in xl.sheet_names if re.match(r"^NBT\s*\d+[, ]\s*Q\d+$", s.strip(), re.IGNORECASE)), None)
    if not nbt_sheet:
        raise ValueError("❌ NBT sheet like 'NBT 9 Q2' not found.")

    df = xl.parse(nbt_sheet, dtype=str)
    df = standardize_columns(df)

    # Create a mapping for possible variants
    col_map = {}
    for col in df.columns:
        if col in ['policyno', 'policynumber', 'policynum', 'policy_no','policyno.']:
            col_map['policyno'] = col
        if col == 'levelcode':
            col_map['levelcode'] = col
        if col == 'estimatedapi':
            col_map['estimatedapi'] = col
        if col in ['count', 'lives']:
            col_map['count'] = col

    required_cols = ['policyno', 'levelcode', 'estimatedapi', 'count']
    for req in required_cols:
        if req not in col_map:
            raise ValueError(f"Missing expected column: {req} in NBT file")

    df = df.rename(columns={v: k for k, v in col_map.items()})

    df = df.dropna(subset=['policyno', 'levelcode'])
    df['levelcode'] = df['levelcode'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str[-8:].str.zfill(8)
    df['estimatedapi'] = pd.to_numeric(df['estimatedapi'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    df['count'] = pd.to_numeric(df['count'].astype(str).str.replace(r'[^\d]', '', regex=True), errors='coerce')

    if (df['count'] % 10 == 0).all() and (df['count'] > 0).any():
        df['count'] = df['count'] / 10

    df = df.dropna(subset=['estimatedapi', 'count'])
    return df[df['levelcode'].str.len() == 8]

def extract_high_api_bonus(nbt_df, master_df):
    """Extract high API policies and calculate monthly premium.
    
    Args:
        nbt_df: Cleaned NBT DataFrame
        master_df: Cleaned master agent DataFrame
        
    Returns:
        DataFrame with high API policies (monthly premium) and bonuses
    """
    merged = nbt_df.merge(master_df, on='levelcode', how='inner')
    high_api = merged[merged['estimatedapi'] >= 600000].copy()
    
    # Convert annual premium to monthly (divide by 12)
    high_api['monthlypremium'] = high_api['estimatedapi'] / 12
    high_api['bonus'] = 5000
    
    return high_api[[
        'levelcode', 
        'agentname', 
        'agency', 
        'tenure', 
        'policyno', 
        'monthlypremium',  # Changed from estimatedapi
        'bonus'
    ]].sort_values(by='monthlypremium', ascending=False)

def calculate_agent_rewards(nbt_df, master_df):
    filtered_df = nbt_df[nbt_df['estimatedapi'] < 600000]
    merged = filtered_df.merge(master_df, on='levelcode', how='inner')
    grouped = merged.groupby(['levelcode', 'agentname', 'agency', 'tenure']).agg(estimatedapi=('estimatedapi', 'sum'), lives=('count', 'sum')).reset_index()
    conditions = [
        (grouped['tenure'] < 3) & (grouped['estimatedapi'] >= 42000) & (grouped['estimatedapi'] < 72000),
        (grouped['estimatedapi'] >= 360000),
        (grouped['estimatedapi'] >= 240000),
        (grouped['estimatedapi'] >= 180000),
        (grouped['estimatedapi'] >= 120000),
        (grouped['estimatedapi'] >= 72000)
    ]
    rewards = [1000, 4000, 3000, 2000, 1500, 1000]
    grouped['reward'] = np.select(conditions, rewards, default=0)
    return grouped.sort_values(by='reward', ascending=False)

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

def get_unit_leader_target_and_reward(agency_name):
    if not isinstance(agency_name, str):
        return (np.nan, np.nan)
    agency_upper = agency_name.upper().strip()
    if any(loc in agency_upper for loc in ["MERU", "NANYUKI", "KITUI", "EMBU"]):
        return (100000, 2000)
    elif any(loc in agency_upper for loc in ["MOMBASA", "NAKURU", "KISII", "THIKA", "ELDORET", "KISUMU"]):
        return (250000, 4000)
    elif "NAIROBI" in agency_upper:
        return (500000, 6000)
    return (np.nan, np.nan)

def calculate_unit_leader_rewards(nbt_df, agency_unit_file, active_agents_file):
    xl = pd.ExcelFile(agency_unit_file)
    normalized_sheets = {s.strip().lower(): s for s in xl.sheet_names}
    if 'unit' not in normalized_sheets:
        raise ValueError("❌ Sheet 'Unit' not found.")
    unit_df = xl.parse(normalized_sheets['unit'], dtype=str)
    unit_df = standardize_columns(unit_df)
    if unit_df.columns[0] == 'unnamed:0':
        unit_df = unit_df.iloc[:, 1:]
    unit_df = unit_df[['ulcode', 'unitcode', 'name']]
    unit_df.columns = ['ulcode', 'unitcode', 'leader']

    agents_df = pd.read_excel(active_agents_file, sheet_name=0, dtype=str)
    agents_df = standardize_columns(agents_df)
    agents_df = agents_df[['agentcode', 'unitcode', 'agency']]
    agents_df.columns = ['levelcode', 'unitcode', 'agency']
    agents_df['levelcode'] = agents_df['levelcode'].str.strip().str.zfill(8)

    merged = nbt_df.merge(agents_df, on='levelcode', how='inner')
    unit_agg = merged.groupby(['unitcode', 'agency']).agg(achievedapi=('estimatedapi', 'sum')).reset_index()
    full_df = unit_agg.merge(unit_df, on='unitcode', how='inner')

    full_df[['weeklytarget', 'basereward']] = full_df['agency'].apply(lambda x: pd.Series(get_unit_leader_target_and_reward(x)))
    full_df['percentage'] = (full_df['achievedapi'] / full_df['weeklytarget']).round(2)
    full_df['reward'] = full_df.apply(lambda row: row['basereward'] if row['percentage'] >= 0.9 and row['achievedapi'] >= row['weeklytarget'] else 0, axis=1)
    return full_df[['unitcode', 'leader', 'agency', 'weeklytarget', 'achievedapi', 'percentage', 'reward']].sort_values(by='reward', ascending=False)

def calculate_agency_rewards(nbt_df, master_df):
    merged = nbt_df.merge(master_df[['levelcode', 'agency']], on='levelcode', how='inner')
    agency_grouped = merged.groupby('agency').agg(achievedapi=('estimatedapi', 'sum')).reset_index()
    agency_grouped[['weeklytarget', 'rewardbase']] = agency_grouped['agency'].apply(lambda x: pd.Series(get_agency_target_and_reward(x)))
    agency_grouped['percentage'] = (agency_grouped['achievedapi'] / agency_grouped['weeklytarget']).round(2)
    agency_grouped['reward'] = agency_grouped.apply(lambda row: row['rewardbase'] if row['percentage'] >= 0.9 and row['achievedapi'] >= row['weeklytarget'] else 0, axis=1)
    return agency_grouped[['agency', 'weeklytarget', 'achievedapi', 'percentage', 'reward']].sort_values(by='reward', ascending=False)

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
    xl = pd.ExcelFile(nbt_file)
    sheet_name = next((s for s in xl.sheet_names if re.match(r"^NBT\s*\d+[, ]\s*Q\d+$", s.strip(), re.IGNORECASE)), None)
    if not sheet_name:
        raise ValueError("❌ NBT sheet like 'NBT 9 Q2' not found.")

    cleaned_sheet_name = sheet_name.strip().replace(",", "").replace("  ", " ")
    filename = f"{cleaned_sheet_name} Incentive Report.xlsx"

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
        format_excel_sheet(writer, 'Unit Leader Reward', unit_leader_df, percent_cols=['percentage'])
        format_excel_sheet(writer, 'Agency Reward', agency_reward_df, percent_cols=['percentage'])

    output.seek(0)
    return output, filename
