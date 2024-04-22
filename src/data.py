
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class FTRC_Data:
    '''
    Class for loading the battery failure databank excel spreadsheet into
    dataframes, with various methods for easily handling common data processing needs.
    '''

    def __init__(self, file_path='data/BatteryFailureDatabankV2.xlsx', sheet_name_calorimetry='Fractional-Calorimetry-Data', sheet_name_cell_characteristics='Cell-Characteristics'):
        self.file_path = Path(file_path)
        self.sheet_name_calorimetry = sheet_name_calorimetry
        self.sheet_name_cell_characteristics = sheet_name_cell_characteristics

        # load the data set
        df = pd.read_excel(self.file_path, sheet_name=sheet_name_calorimetry, header=2, usecols=range(1,66))
        # Several columns/rows use '-' instead of just being blank to denote missing data. Force numeric values.
        idx_numeric_cols = [8,9,10,11,12,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,54,55,56,57,58,59,60,61,62,63,64]
        for idx_col in idx_numeric_cols:
            df[df.columns[idx_col]] = df[df.columns[idx_col]].apply(pd.to_numeric, errors='coerce').astype(float)
        # Write df property
        self.df = df

        # Perform repeated data processing/loading operations
        self.process_battery_failure_databank()        
        
    def process_battery_failure_databank(self): 
        """
        Most common changes to data:
            - adding columns for Total, Pos, Neg Mass Ejected 
            - append cell capacities for calorimetry dataframe from cell characteristics
            - normalizing ejected mass fractions (by cell mass) and heat ouputs (by cell charge capacity)
            - 
        """
        self.sum_ejected_masses()
        self.append_cell_capacity()
        self.normalize_ejected_mass_and_heat()
        self.append_geometry()
        self.append_manufacturer()

        # Logical feature denoting if the bottom vent was actuated during thermal runaway
        self.df.loc[:, 'BV Actuated'] = True
        self.df.loc[(self.df['Cell-Failure-Mechanism']=='Top Vent') | (self.df['Cell-Failure-Mechanism']=='Top Vent Only - Bottom Vent Not Actuated'), 'BV Actuated'] = False
        
    def sum_ejected_masses(self):
        df = self.df.copy()
        # total mass
        total_mass = (
            df["Post-Test-Mass-Positive-Ejecta-Mating-g"] 
            + df["Post-Test-Mass-Positive-Ejecta-Bore-Baffles-g"] 
            + df["Post-Test-Mass-Positive-Copper-Mesh-g"] 
            + df["Post-Test-Mass-Negative-Ejecta-Mating-g"] 
            + df["Post-Test-Mass-Negative-Ejecta-Bore-Baffles-g"] 
            + df["Post-Test-Mass-Negative-Copper-Mesh-g"] 
            + df["Post-Test-Mass-Unrecovered-g"]
        )
        df.loc[:,'Total-Mass-Ejected-g'] = total_mass
        # positive mass ejected
        df.loc[:,'Positive-Mass-Ejected-g'] = (
            df["Post-Test-Mass-Positive-Ejecta-Mating-g"] 
            + df["Post-Test-Mass-Positive-Ejecta-Bore-Baffles-g"] 
            + df["Post-Test-Mass-Positive-Copper-Mesh-g"]
        )
        # negative mass ejected
        df.loc[:,'Negative-Mass-Ejected-g'] = (
            df["Post-Test-Mass-Negative-Ejecta-Mating-g"] 
            + df["Post-Test-Mass-Negative-Ejecta-Bore-Baffles-g"] 
            + df["Post-Test-Mass-Negative-Copper-Mesh-g"]
        )

        self.df = df

    def append_cell_capacity(self): 
        """Reads in cell capacity from cell info sheet and adds cell capacity column according to cell desc."""
        cell_characteristics = pd.read_excel(self.file_path, sheet_name=self.sheet_name_cell_characteristics, header=2, usecols=range(1,17))
        is_nan_capacity = np.isnan(cell_characteristics['Cell-Capacity-Ah'])
        cell_characteristics = cell_characteristics.loc[~is_nan_capacity]

        cd_list = cell_characteristics['Cell-Description'].to_list()
        cc_list = cell_characteristics['Cell-Capacity-Ah'].to_list()

        cell_dict = {}

        for key, val in zip(cd_list, cc_list):
             cell_dict[key] = val

        cell_cap_list = self.df['Cell-Description'].to_list()

        final_col_list = []
        for cell_cap in cell_cap_list:
            final_col_list.append(cell_dict[cell_cap])

        self.df['Cell-Capacity-Ah'] = final_col_list
        
    def normalize_ejected_mass_and_heat(self):
        """Normalizing values, dividing mass ejections by pre test cell mass and energy kJ by cell capacity."""
        self.df.loc[:,'Total Ejected Mass Fraction [g/g]'] = self.df['Total-Mass-Ejected-g'] / self.df['Pre-Test-Cell-Mass-g']
        self.df.loc[:,'Total Heat Output [kJ/A*h]'] = self.df['Corrected-Total-Energy-Yield-kJ'] / self.df['Cell-Capacity-Ah']
        
        self.df.loc[:,'Unrecovered Mass Fraction [g/g]'] = self.df['Post-Test-Mass-Unrecovered-g'] / self.df['Pre-Test-Cell-Mass-g']
        self.df.loc[:,'Body Mass Remaining Fraction [g/g]'] = self.df['Post-Test-Mass-Cell-Body-g'] / self.df['Pre-Test-Cell-Mass-g']
        self.df.loc[:,'Positive Ejected Mass Fraction [g/g]'] = self.df['Positive-Mass-Ejected-g'] / self.df['Pre-Test-Cell-Mass-g']
        self.df.loc[:,'Negative Ejected Mass Fraction [g/g]'] = self.df['Negative-Mass-Ejected-g'] / self.df['Pre-Test-Cell-Mass-g']

        self.df.loc[:,'Cell Body Heat Output [kJ/A*h]'] = self.df['Energy-Fraction-Cell-Body-kJ'] / self.df['Cell-Capacity-Ah']
        self.df.loc[:,'Positive Heat Output [kJ/A*h]'] = self.df['Energy-Fraction-Positive-Ejecta-kJ'] / self.df['Cell-Capacity-Ah']
        self.df.loc[:,'Negative Heat Output [kJ/A*h]'] = self.df['Energy-Fraction-Negative-Ejecta-kJ'] / self.df['Cell-Capacity-Ah']

    def append_geometry(self):
        '''Extract cell geometry from the cell description'''
        is_18650 = ['18650' in cell_descr for cell_descr in self.df['Cell-Description']]
        self.df.loc[is_18650, 'Geometry'] = int(18650)
        is_21700 = ['21700' in cell_descr for cell_descr in self.df['Cell-Description']]
        self.df.loc[is_21700, 'Geometry'] = int(21700)
        is_33600 = ['VES16' in cell_descr for cell_descr in self.df['Cell-Description']]
        self.df.loc[is_33600, 'Geometry'] = int(33600)

    def append_manufacturer(self):
        '''Extract manufacturer from the cell description'''
        manufacturers = [
            'KULR',
            'LG',
            'MOLiCEL',
            'Panasonic',
            'Saft',
            'Samsung',
            'Sanyo',
            'Sony',
            'Soteria',
        ]
        for manufacturer in manufacturers:
            is_manufacturer = [manufacturer in cell_descr for cell_descr in self.df['Cell-Description']]
            self.df.loc[is_manufacturer, 'Manufacturer'] = manufacturer
        
    def remove(self, rows_to_remove, col_name="Cell-Description"):
        """ removes column from pandas dataframe """
        self.df = self.df[self.df[col_name] != rows_to_remove]

class Split():
    '''One-hot encodes meta data columns, splits train and test, separates feature and target columns'''
    def __init__(self,
                 data,
                 split_method='cell_type_split',
                 cell_type_test = None,
                 one_hot_columns=['Cell-Description', 'Manufacturer' , 'Geometry', 'Trigger-Mechanism', 'Cell-Failure-Mechanism']
                 ):
        self.data = data
        self.split_method = split_method # 'cell_type_split' or '80_20_split'
        if split_method == 'cell_type_split':
            if cell_type_test is None:
                raise ValueError("'cell_type_test' cannot be None if split is by cell_type.")
        self.one_hot_encode(columns=one_hot_columns)
        self.split_cells(cell_type_test)

    def get_splits(self, is_numpy=True):
        if is_numpy:
            return self.x_train.to_numpy(), self.y_train.to_numpy(), self.x_test.to_numpy(), self.y_test.to_numpy()
        else:
            return self.x_train, self.y_train, self.x_test, self.y_test
    
    def one_hot_encode(self, columns):
        """One hot encodes the columns in list: column"""
        for column in columns:
            if np.any(self.data.columns == column):
                to_encode = pd.get_dummies(self.data[column])
                self.data = pd.concat((self.data, to_encode), axis=1)
                self.data = self.data.drop(column, axis=1)
        
    def split_cells(self, cell_type_test):
        """Splits cells either by 80/20 split or by cell defined by cell_type"""
        y_columns = [
                'Total Heat Output [kJ/A*h]', 
                'Cell Body Heat Output [kJ/A*h]',
                'Positive Heat Output [kJ/A*h]',
                'Negative Heat Output [kJ/A*h]',
                ]
        if self.split_method == '80_20_split':
            x = self.data.drop(y_columns, axis=1).copy()
            y = self.data.loc[:,y_columns]
            
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, shuffle=True, random_state=42, test_size=0.2)
            
        elif self.split_method == 'cell_type_split': 
            data_test = self.data[self.data[cell_type_test] == 1]
            data_train = self.data[self.data[cell_type_test] == 0]

            self.y_test = data_test.filter(y_columns)
            self.x_test = data_test.drop(self.y_test, axis=1)

            self.y_train = data_train.filter(y_columns)
            self.x_train = data_train.drop(self.y_train, axis=1)
        else:
            raise ValueError("'split_method' must be 'cell_type_split' or '80_20_split'")