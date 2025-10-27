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
├── docs/
│   ├── data_dictionary.md                   # Abbreviations etc...
│
├── notebooks/
│   ├── 00_ElectricityDemand_DomainResearch.ipynb    # Contains fundamental domain knowledge
│   ├── 01_OPSD_EDA.ipynb                                           # First look at the OPSD data set
│   ├── 02_HolidaysAndSchoolFreeDays.ipynb               # Relevant German holiday data
│   └── 03_Weather_historic.ipynb                                   # German weather data
│   
│
├── src/
│   ├── config.py
│   ├── de_hol_df.py
│   ├── dwd_utils.py                                  # download helper and DataFrame builders
│   ├── eda_utils.py                                   # missingness summaries
│   ├── load_data.py
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

---

##  Data Source and Usage Terms

### Open Power System Data (OPSD)

This project uses electricity demand and generation data from **Open Power System Data (OPSD)**, a public data platform curated by the OPSD project team.
The data are accessible at [https://open-power-system-data.org/](https://open-power-system-data.org/).

All OPSD datasets are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. Users who download or process this data through the routines included in this repository are **responsible for complying with that license**, including **attributing Open Power System Data (OPSD)** as the original data provider.

### Deutscher Wetterdienst (DWD)

This project uses publicly available meteorological data provided by the **Deutscher Wetterdienst (DWD)** — Germany’s national meteorological service — through its [Open Data Portal](https://opendata.dwd.de/climate_environment/CDC/).

All DWD data are subject to the [official DWD Open Data terms of use](https://www.dwd.de/EN/service/copyright/copyright_artikel.html). Users who download or process this data through the routines included in this repository are **responsible for ensuring compliance with those terms**.


## 🪶 License

This project is for **educational and non-commercial use**.  
As already stated above, all data sources remain under their respective open licenses (OPSD, DWD).

---

