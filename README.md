#  Electricity Demand Forecast Project

A data-science project analyzing and forecasting electricity demand in Germany. First step, training prediction models, based on the historical (2015-2020) **Open Power System Data (OPSD)** dataset, combined with  **German holiday data** and **weather data**.

---

## Project Structure (as of now)

```
electricity-demand-project/
│
├── data/
│   ├── raw/                                           # Original input data (OPSD, DWD, etc.)
│   ├── processed/                                 #  OPSD data set and German holiday data as parquet
│   └── external/                                    # Optional external sources (e.g., UBA, weather APIs)
│
├── notebooks/
│   ├── 00_ElectricityDemand_DomainResearch.ipynb    # Contains fundamental domain knowledge
│   ├── 01_OPSD_EDA.ipynb                                           # First look at the OPSD data set
│   └── 02_HolidaysAndSchoolFreeDays.ipynb               # Relevant German holiday data
│   
│
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── eda_utils.py
│   ├── plotting_utils.py
│   └── data_modules/
│       └── german_school_semesters.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Sources

- **Electricity load**: [Open Power System Data – Time Series 60min](https://data.open-power-system-data.org/time_series/)
- **Weather data**: Deutscher Wetterdienst (DWD) historical hourly data picked from a set of different weather stations
- **Holiday data**: Manually structured German school holiday dictionaries

---

## Main DataFrames

- `OPSD_60min_df`: full OPSD time series
- `OPSD_60min_de_lu_df`: Subframe of Germany and Luxembourg related Columns
- `DE_hol_df`: national holiday/school-free calendar (daily)



## 🪶 License

This project is for **educational and non-commercial use**.  
All data sources remain under their respective open licenses (OPSD, DWD).

---

