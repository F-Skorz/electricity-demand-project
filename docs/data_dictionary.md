<h1> Data Dictionary </h1>
<h2>  Comprehensive Abbreviations Dictionary </h2>

| Abbreviation  | Un-abbreviated Form | Explanation |
| --------------|---------------------------|---------------------|
| `AMP` | Amprion GmbH, `amprion` | One of Germany's four **TSO**s |
| `BW`| Baden-Württemberg |  Germany's third state in terms of population, the only one with a **TSO** of its own: **TransnetBW** |
| `BY`| Bavaria (Bayern) |  |
| `BE` |  Berlin | The capital of Germany and a federal (city) state, part of **50Hertz**' control area  |
| `BB` |     Brandenburg |  | 
| `CEST` | Central European Summer Time | The daylight-saving version of Central European Time (**UTC + 2**) |
|  `CET` | Central European Time | The standard time zone used in most of Central Europe (**UTC + 1**) | 
| ENTSO-E | European Network of Transmission System Operators for Electricity |  The association of Europe’s electricity transmission system operators (**TSO**s). |
| DST  | Daylight Saving Time | The practice of advancing clocks during warmer months to extend evening daylight |
| `DE` | Germany |    |
| **DWD** | Deutscher Wetterdienst | Germany’s national meteorological service |
|`HB`| Hansestadt Bremen |  The smallest of Germany's three city states |
|`HH`|  Hansestadt Hamburg|  |
|`HE`| Hesse (Hessen) |   |
| `MV` | Mecklenburg-Vorpommern |  |
|`NI`| Lower Saxony (Niedersachsen)       |  |
|`NW`|  North Rhine-Westphalia (Nordrhein-Westfalen) | Biggest federal State of Germany, population-wise |
| OPSD | Open Power System Data | A project providing open, high-quality data for electricity system analysis |
| QN_x | Qualitätsniveau |  **DWD** quality codes — numeric flags (x = dataset number) indicating the data quality or processing level (Qualitätsniveau = quality level) |
| QN_8 | Qualitätsniveau |  **DWD** quality code with respect to precipitation data |
| QN_9 | Qualitätsniveau |  **DWD** quality code with respect to temperature and humidity data |
|`RP`| Rhineland-Palatinate (Rheinland-Pfalz)|  | 
| `RH` | Relative Humidity | The percentage of water vapor in the air compared to the maximum it could hold at that temperature |
|`SL` | Saarland | Smallest non-city state  and unit of measurement |
| TSO | Transmission System Operator |  Companies that run the high-voltage electricity grid | 
| `SN`| Saxony (Sachsen) | | 
|`ST`| Saxony-Anhalt (Sachsen-Anhalt) |   |
|`SH`| Schleswig-Holstein |   |
|`TEN`| Tennet TSO GmbH   |  One of Germany's four **TSO**s  |
|`TH`| Thuringia (Thüringen) |   |
|`TNB`|  **TransnetBW  GmbH**, `transnetbw` |  One of Germany's four **TSO**s, **TransnetBW GmbH** operates the electricity transmission system in Baden-Württemberg |
| `UTC` | Coordinated Universal Time | The global standard time reference that does not change with time zones or daylight saving | 
|`50H` | 50Hertz Transmission GmbH | One of Germany's four **TSO**s,  responsible for the high power network in eastern Germany as well as the area around Hamburg

<h2>  Abbreviations for Germany's Federal States</h2>

We use the following abbreviations for the German federal states.

| Abbreviation  | State  |  Population (m, as of 2025) |
| --------------|---------|----------------------:|
| `BW`| Baden-Württemberg |  11.11 |
 `BY`| Bavaria (Bayern) | 13.44 |
 | `BE` |  Berlin |  3.90 |
 | `BB` |     Brandenburg | 2.56 | 
 |`HB`| Hansestadt Bremen | 0.71 |
 |`HH`|  Hansestadt Hamburg| 1.96 |
 |`HE`| Hesse (Hessen) |  6.43 |
 | `MV` | Mecklenburg-Vorpommern | 1.63 |
 |`NI`| Lower Saxony (Niedersachsen)       | 8.16 |
 |`NW`|  North Rhine-Westphalia (Nordrhein-Westfalen) | 18.15 |
 |`RP`| Rhineland-Palatinate (Rheinland-Pfalz)| 4.17 | 
 |`SL` | Saarland | 0.99 |
 | `SN`| Saxony (Sachsen) | 4.09 |
 |`ST`| Saxony-Anhalt (Sachsen-Anhalt) | 2.14 |
 |`SH`| Schleswig-Holstein | 3.00 |
 |`TH`| Thuringia (Thüringen) | 2.12 |


<h2> The Weather Data </h2>

We retrieve weather data for the following German cities.


| City      |  City ID               |   State   | Latitude   | Longitude  | TSO   |
|--------- |------------------ |---------|-------------|------------ |---------|
|  Berlin   |  `DE_BE_50H_1`  | Berlin  |  52.5200	    | 13.4050	| 50Hertz |
|  Hamburg |   `DE_HH_50H_1`  | Hamburg  |  53.5511 |	9.9937	| 50Hertz |
| Frankfurt | `DE_HE_AMP_1` |	Hesse (Hessen)  | 50.1109	| 8.6821	| Amprion |
| Cologne (Köln) | `DE_NW_AMP_1` |  North Rhine-Westphalia (Nordrhein-Westfalen) |	50.9375	| 6.9603 |	Amprion |
| Munich (München) | `DE_BY_TEN_1` |	Bavaria (Bayern) |	48.1351 |	11.5820 |	TenneT | 
| Augsburg | `DE_BY_AMP_1` |	Bavaria (Bayern) |	48.3705 |	10.8978 |	Amprion |
| Stuttgart	| `DE_BW_TNB_1` |	 Baden-Württemberg	| 48.7758	| 9.1829 |	TransnetBW |
| Freiburg |`DE_BW_TNB_2` | Baden-Württemberg | 47.9990	| 7.8421 |	TransnetBW |
| Leipzig |	`DE_SN_50H_1`	| Saxony (Sachsen) | 51.3397 |	12.3731 |	50Hertz |
| Dresden | `DE_SN_50H_2` | Saxony (Sachsen)	| 51.0504 |	13.7373 |	50Hertz| 
|	Kiesl | `DE_SH_TEN_1`	|	Schleswig-Holstein	| 54.3233	| 10.1228	| TenneT |

and, there, from these **DWD** weather stations

| City       | Station ID | Station Name             | 
|------------|------------|--------------------------|
| Berlin     | 00403      | Berlin-Dahlem (FU)       | 
| Hamburg    | 01975      | Hamburg-Fuhlsbüttel      |
| Frankfurt  | 01420      | Frankfurt/Main           |  
| Cologne    | 02667      | Köln/Bonn                | 
| Augsburg   | 00232      | Augsburg                 | 
| Munich     | 03379      | München-Stadt            | 
| Stuttgart  | 04928      | Stuttgart (Schnarrenberg)|             
| Freiburg   | 01443      | Freiburg                 | 
| Leipzig    | 02932      | Leipzig/Halle            | 
| Kiel       | 02564      | Kiel-Holtenau            | 
| Dresden    | 01048      | Dresden-Klotzsche        | 
