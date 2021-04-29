This REGRESSIONreadme.txt file was generated on 2020-06-19 by A SYTSMA

GENERAL INFORMATION

1. Title of Dataset: Biomass-Remote Sensing Regression Data

2. Author Information
	A. Principal Investigator Contact Information
		Name: Aidan Chicchetti
		Institution: UC Berkeley
		Email: achecchetti@berkeley.edu


	B. Associate or Co-investigator Contact Information
		Name: Anneliese Sytsma
		Institution: UC Berkeley
		Email: anneliese_sytsma@berkeley.edu



3. Date of data collection (single date, range, approximate date): multiple dates 2017-04-10, 2019-06-12, 2019-01-24, 2019,02-28

4. Geographic location of data collection: Oro Loma Treatment Wetland and Horizontal Levee, San Lorenzo, CA, US

5. Information about funding sources that supported the collection of the data: NSF ERC Reinventing the Nations Urban Water Infrastructure (ReNUWIt)


SHARING/ACCESS INFORMATION


1. Licenses/restrictions placed on the data: The PlanetScope data is licensed to the authors for non-commercial research purposes at no cost via the Planet Education and Research Program.
	For this reason, the Planet Data is not included in this Mendeley Dataset. Anyone interested in downloading the data can do so through Planet Team.
	Planet Team (2017). Planet Application Program Interface: In Space for Life on Earth. San Francisco, CA. https://api.planet.com.

2. Links to publications that cite or use the data: doc:10.1016/j.wroa.2020.100070

3. Links to other publicly accessible locations of the data: doi:10.17632/xwx83vzmf6.2

4. Was data derived from another source? yes
	A. sr_thirds: derived from PlanetScope Scene imagery, 4-band, 3.7 m spectral resolution (Planet Team, 2017). 
	B. ndvi_thirds: derived from PlanetScope Scene imagery, 4-band, 3.7 m spectral resolution (Planet Team, 2017). 
	C. gndvi_thirds: derived from PlanetScope Scene imagery, 4-band, 3.7 m spectral resolution (Planet Team, 2017). 

6. Recommended citation for this dataset: 

Cecchetti, Aidan; Stiegler, Angela; Sytsma, Anneliese; Gonthier, Emily; Graham, Katherine; Boehm, Alexandria B.; Dawson, Todd; Sedlak, David (2020), “Horizontal Levee Monitoring Data”, Mendeley Data, v2
http://dx.doi.org/10.17632/xwx83vzmf6.2

DATA & FILE OVERVIEW

1. Folder/file List: 
	A. sr_thirds (folder): contains .csv files with GNDVI index values summarized by cell-third for each biomass sampling date and isotope sampling date
	B. ndvi_thirds (folder): contains .csv files with NDVI index values summarized by cell-third for each biomass sampling date and isotope sampling date
	C. gndvi_thirds (folder): contains .csv files with SR index values summarized by cell-third for each biomass sampling date and isotope sampling date
	D. biomass_reg.ipynb: python notebook with functions used to develop biomass-SR regression
	E. biomass_py.csv: biomass measurements (in kg-m2) by cell-third and date

2. Relationship between files, if important: 
	A. biomass_reg.ipynb references sr_thirds, ndvi_thirds, and gndvi_thirds folders and biomass_py.csv file. 

3. Additional related data collected that was not included in the current data package: n/a

4. Are there multiple versions of the dataset? no


METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data: 
	A. Generating biomass-vegetation index relationship:
		i. Read in vegetation index .csv files to python
		ii. Read in biomass_py.csv to python
		ii. Conduct linearity analysis, OLS regression

2. Instrument- or software-specific information needed to interpret the data: 
	A. To open ArcGIS files, ArcMap 10.7 is needed.
	B. To open and run the biomass_reg.ipynb, python3.6 and jupyter notebook are needed.
	C. To open .csv files, a text reader or ms excel is needed.
________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________________
DATA-SPECIFIC INFORMATION FOR: [gndvi_thirds] [FOLDER]

1. Number of variables: 2

2. Number of cases/rows: 36

3. Variable List: 

	variable 1: Cell_Third, 	description: label of wetland cell third, 		units: n/a
	variable 2: MEAN, 		description: mean GNDVI value within cell-third, 	units: n/a

4. Specialized formats or other abbreviations used: name of each .csv file contains date that correponds to PlanetScope image date.
________________________________________________________________________________________________________________________________
DATA-SPECIFIC INFORMATION FOR: [ndvi_thirds] [FOLDER]

1. Number of variables: 2

2. Number of cases/rows: 36

3. Variable List: 

	variable 1: Cell_Third, 	description: label of wetland cell third, 			units: n/a
	variable 2: MEAN, 		description: mean NDVI value within cell-third, 		units: n/a

4. Specialized formats or other abbreviations used: name of each .csv file contains date that correponds to PlanetScope image date.
________________________________________________________________________________________________________________________________
DATA-SPECIFIC INFORMATION FOR: [sr_thirds] [FOLDER]
 
1. Number of variables: 2

2. Number of cases/rows: 36

3. Variable List: 

	variable 1: Cell_Third, 	description: label of wetland cell third, 			units: n/a
	variable 2: MEAN, 		description: mean SR value within cell-third, 			units: n/a

4. Specialized formats or other abbreviations used: name of each .csv file contains date that correponds to PlanetScope image date.
________________________________________________________________________________________________________________________________
DATA-SPECIFIC INFORMATION FOR: [biomass_py.csv] [FILE]
 
1. Number of variables: 4

2. Number of cases/rows: 37

3. Variable List: 

	variable 1: Cell_Third,		description: label of wetland cell third, 			units: n/a
	variable 2: MEAN_Bioma, 	description: mean biomass value within cell-third, 		units: kg/m2
	variable 3: Date_planet, 	description: PlanetScope image date, 				units: n/a
	variable 2: Date, 		description: Sample date,			 		units: n/a

4. Specialized formats or other abbreviations used: n/a
________________________________________________________________________________________________________________________________
DATA-SPECIFIC INFORMATION FOR: [core_reg.csv] [FILE]
 
1. Number of variables: 9

2. Number of cases/rows: 36

3. Variable List: 

	variable 1: FID, 		description: index (unique ID) for sample-biomass regression, 	units: n/a
	variable 2: year,		description: Sample year,			 		units: n/a
	variable 3: Date_planet, 	description: PlanetScope image date, 				units: n/a
	variable 4: GNDVI,		description: mean GNVI value within cell-third, 		units: n/a
	variable 5: NDVI, 		description: mean NDVI value within cell-third, 		units: n/a
	variable 6: SR,		 	description: mean SR value within cell-third, 			units: n/a
	variable 7: MEAN_Bioma, 	description: mean biomass value within cell-third, 		units: kg/m2
	variable 8: Cell_Third,		description: label of wetland cell third, 			units: n/a
	variable 9: Cell,		description: label of wetland cell,				units: n/a

4. Specialized formats or other abbreviations used: n/a
________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________________