import pandas as pd
import numpy as np

def calculate_speed(x_component, y_component):
    """Calculate speed from x and y components"""
    return np.sqrt(x_component**2 + y_component**2)

def calculate_absolute_direction(x_component, y_component):
    """Calculate absolute direction in degrees from x and y components"""
    return (np.arctan2(x_component, y_component) * (180 / np.pi)) % 360

def calculate_relative_direction(absolute_direction, course):
    """Calculate relative direction, ensuring it's between 0 and 180 degrees"""
    relative_dir = abs(absolute_direction - course)
    return relative_dir.where(relative_dir <= 180, 360 - relative_dir)

def process_environmental_data(fusion_data):
    """Process environmental data including wind, wave, and current data"""
    data = pd.DataFrame()
    
    # Basic navigation data
    data["Time"] = fusion_data["time"]
    data["Longitude"] = fusion_data["Longitude"]
    data["Latitude"] = fusion_data["Latitude"]
    data["ShipSpeed"] = fusion_data["sp"]
    data["Course"] = fusion_data["co"]
    
    # Wind data processing
    wind_u = fusion_data["10 metre U wind component"]
    wind_v = fusion_data["10 metre V wind component"]
    data["WindSpeed"] = calculate_speed(wind_u, wind_v)
    absolute_wind_direction = calculate_absolute_direction(wind_u, wind_v)
    data["RelativeWindDirection"] = calculate_relative_direction(absolute_wind_direction, data["Course"])
    
    # Wave data processing
    data["WaveHeight"] = fusion_data["Significant height of combined wind waves and swell"]
    data["WavePeriod"] = fusion_data["Mean wave period"]
    data["RelativeWaveDirection"] = calculate_relative_direction(
        fusion_data["Mean wave direction"], 
        data["Course"]
    )
    
    # Current data processing
    current_u = fusion_data["Eastward sea water velocity"]
    current_v = fusion_data["Northward sea water velocity"]
    data["CurrentSpeed"] = calculate_speed(current_u, current_v)
    absolute_current_direction = calculate_absolute_direction(current_u, current_v)
    data["RelativeCurrentDirection"] = calculate_relative_direction(
        absolute_current_direction, 
        data["Course"]
    )
    
    return data

def main():
    """Main function to process data for all ships"""
    mmsis = ["215126000", "215131000", "215189000", "215240000", "215577000"]
    
    for mmsi in mmsis:
        # Read raw data
        fusion_data = pd.read_excel(f"./data/{mmsi}_ow_oc_all.xlsx")
        
        # Process data
        processed_data = process_environmental_data(fusion_data)
        
        # Save processed data
        processed_data.to_excel(f"./fusion_data/target_{mmsi}.xlsx", index=False)
        print(f"Processed data saved for MMSI: {mmsi}")

if __name__ == "__main__":
    main()
    