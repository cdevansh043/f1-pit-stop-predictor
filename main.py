import fastf1
from pathlib import Path 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

main_dir = Path(__file__).resolve().parent
data_file = main_dir / "f1_pit_stop_data.csv"

if not data_file.exists():
    FASTF1_CACHE = main_dir / 'cache'
    FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(FASTF1_CACHE)


    races_to_collect = [
        # 2023 Season
        (2023, 'Bahrain'), (2023, 'Saudi Arabia'), (2023, 'Australia'), (2023, 'Azerbaijan'), 
        (2023, 'Miami'), (2023, 'Monaco'), (2023, 'Spain'), (2023, 'Canada'), 
        (2023, 'Austria'), (2023, 'Great Britain'), (2023, 'Hungary'), (2023, 'Belgium'), 
        (2023, 'Netherlands'), (2023, 'Italy'), (2023, 'Singapore'), (2023, 'Japan'), 
        (2023, 'Qatar'), (2023, 'United States'), (2023, 'Mexico'), (2023, 'Brazil'), 
        (2023, 'Las Vegas'), (2023, 'Abu Dhabi'),
        
        # 2024 Season
        (2024, 'Bahrain'), (2024, 'Saudi Arabia'), (2024, 'Australia'), (2024, 'Japan'), 
        (2024, 'China'), (2024, 'Miami'), (2024, 'Emilia Romagna'), (2024, 'Monaco'), 
        (2024, 'Canada'), (2024, 'Spain'), (2024, 'Austria'), (2024, 'Great Britain'), 
        (2024, 'Hungary'), (2024, 'Belgium'), (2024, 'Netherlands'), (2024, 'Italy'), 
        (2024, 'Azerbaijan'), (2024, 'Singapore'), (2024, 'United States'), (2024, 'Mexico'), 
        (2024, 'Brazil'), (2024, 'Las Vegas'), (2024, 'Qatar'), (2024, 'Abu Dhabi')
    ]

    all_races_data = []
    cols = ['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'PitInTime', 'PitOutTime', 'Position', 'Team']

    print("Starting data collection...")

    for year, race_name in races_to_collect:
        try:
            print(f"Loading {year} {race_name}...")
            
            # Load session
            session = fastf1.get_session(year, race_name, 'R')
            session.load(laps=True, telemetry=False, weather=False)
            
            # Extract and create target variable
            df = session.laps[cols].copy()
            df['is_pit_lap'] = df['PitInTime'].notna().astype(int)
            
            # Add metadata
            df['Year'] = year
            df['RaceName'] = race_name

            # Dropping PitInTime & PitOutTime
            df = df.drop(columns=['PitInTime', 'PitOutTime'])
            # Accumulate
            all_races_data.append(df)
            
        except Exception as e:
            print(f"Skipping {year} {race_name} due to error: {e}")
            continue

    # Concatenate and Save
    if all_races_data:
        final_df = pd.concat(all_races_data, ignore_index=True)
        
        # Save as CSV
        final_df.to_csv('f1_pit_stop_data.csv', index=False)
        print(f"\nSuccess! Saved {len(final_df)} laps to 'f1_pit_stop_data.csv'")
    else:
        print("No data was collected.")

else:
    df = pd.read_csv("f1_pit_stop_data.csv")

    # dropping useless rows
    df = df.dropna(subset=['LapTime', 'TyreLife', 'Compound', 'Position'])
    df = df.reset_index(drop=True)

    # creating lap_time_seconds column
    df['lap_time_seconds'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()

    # sorting values
    df = df.sort_values(by=['Year', 'RaceName', 'Driver', 'LapNumber'], ascending=True)
    # Reset the index
    df = df.reset_index(drop=True)

    # rolling average lap time (tyre degradation proxy)
    df['rolling_avg_lap_time'] = (
    df.groupby(['Year', 'RaceName', 'Driver'])['lap_time_seconds']
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=[0, 1, 2], drop=True)
    )

    # Lap time delta (pace drop signal)
    df['lap_time_delta'] = df['lap_time_seconds'] - df['rolling_avg_lap_time']

    # Tyre age squared
    df['tyre_age_squared'] = df['TyreLife'] ** 2

    # Compound encoding
    compound_map = {
    'SOFT': 0,
    'MEDIUM': 1,
    'HARD': 2,
    'INTERMEDIATE': 3,
    'WET': 4
    }
    df['compound_encoded'] = df['Compound'].map(compound_map)

    # stint_number
    df['stint_number'] = df.groupby(['Year', 'RaceName', 'Driver'])['is_pit_lap'].shift(fill_value=0)
    df['stint_number'] = df.groupby(['Year', 'RaceName', 'Driver'])['stint_number'].cumsum() + 1

    # NOW laps_since_last_pit can use stint_number
    df['laps_since_last_pit'] = (
        df.groupby(['Year', 'RaceName', 'Driver', 'stint_number'])
        .cumcount() + 1
    )

    # race_progress (separate from laps_since_last_pit)
    total_laps = df.groupby(['Year', 'RaceName'])['LapNumber'].max().reset_index()
    total_laps.columns = ['Year', 'RaceName', 'TotalLaps']
    df = df.merge(total_laps, on=['Year', 'RaceName'], how='left')
    df['race_progress'] = df['LapNumber'] / df['TotalLaps']

    # Position features
    df['position_norm'] = df['Position'] / 20.0
    df['position_change'] = df.groupby(['Year', 'RaceName', 'Driver'])['Position'].diff().fillna(0)
    df['is_front_runner'] = (df['Position'] <= 5).astype(int)

    # Categorical encoding
    le = LabelEncoder()
    for col in ['Driver', 'Team', 'RaceName']:
        df[f'{col.lower()}_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))


    # Save the engineered dataset
    df.to_csv('f1_features.csv', index=False)
    print("Feature engineering complete. Saved to 'f1_features.csv'.")
