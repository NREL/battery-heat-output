'''
Trains zero and i-shot models predicting fractional thermal runaway heat output from ejected mass and meta data.
Results from 'dummy' architecture, SVM, and XGBoost.
'''

from src.data import FTRC_Data
from src.modeling import *
import json
from pathlib import Path

if __name__ == "__main__":
    # Import battery failure data bank
    data = FTRC_Data()

    # Remove non-commercial cells, 'test' cells, and cells with less than 10 measurements from the data set
    cells_to_remove = [
        'Soteria 18650 (AL)',
        'Soteria 18650 (ALCU)',
        'Soteria 18650 (CU)',
        'Soteria 18650 (DW)',
        'Soteria 18650 (ALDW)',
        'Soteria 18650 (ALCUDW)',
        'Soteria 18650 (Control)',
        'Saft D-Cell-VES16',
        'MOLiCEL 18650-J',
        'MOLiCEL 18650-M35A',
        'MOLiCEL 18650-P28A',
        'MOLiCEL 18650-Test Cell',
        'MOLiCEL 18650-Test Cell (DW-Gold)',
        'MOLiCEL 18650-Test Cell (DW-Silver)',
        'LG 18650-HG2',
        'LG 18650-M36',
        'LG 18650-Test Cell (NBV-220)',
        'LG 18650-Test Cell (NBV-250)',
        'Panasonic 18650-BE',
        'Samsung 18650-26J',
        'Samsung 18650-30Q',
        'Sony 18650-VTC6',
        ]
    for cell in cells_to_remove:
        data.remove(cell)

    # Defining features to keep
    features_metadata = [
        'Cell-Description',
        'Manufacturer',
        'Geometry',
        'Cell-Capacity-Ah',
        'Trigger-Mechanism',
        'Cell-Failure-Mechanism',
        'BV Actuated',
    ]
    features_ejected_mass = [
        'Total-Mass-Ejected-g', 'Total Ejected Mass Fraction [g/g]', # overall mass loss
        'Post-Test-Mass-Unrecovered-g', 'Unrecovered Mass Fraction [g/g]', # unrecovered mass
        'Pre-Test-Cell-Mass-g', 'Post-Test-Mass-Cell-Body-g', 'Body Mass Remaining Fraction [g/g]', # body mass loss
        'Positive-Mass-Ejected-g', 'Positive Ejected Mass Fraction [g/g]', # positive end mass loss
        'Negative-Mass-Ejected-g', 'Negative Ejected Mass Fraction [g/g]', # negative end mass loss
    ]
    # Target variables
    targets = [
        'Total Heat Output [kJ/A*h]',
        'Cell Body Heat Output [kJ/A*h]',
        'Positive Heat Output [kJ/A*h]',
        'Negative Heat Output [kJ/A*h]',
    ]

    # Lots of extra columns in the data set, only keep the defined subset for modeling
    data_trimmed = data.df[features_metadata + features_ejected_mass + targets]
    
    # Dummy training loop
    print('Dummy training')
    y_test_pred = {}
    errors = {}
    
    cell_types = list(data_trimmed['Cell-Description'].unique())    
    # cell_types = ['KULR 18650-K330'] ### if you want to test quickly for just one cell
    for cell_type in cell_types:
        print(cell_type)
        
        y_test_pred_cell = {}
        errors_cell = {}

        max_i = data_trimmed.value_counts('Cell-Description')[cell_type]
        for iter in range(max_i): # range(max_i):
            print(iter)
            if iter == 0:
                y_test_pred_iter, errors_iter, mass_ejected_test_iter = zero_shot_dummy(data_trimmed, cell_type_test=cell_type, is_chain=False)
            else:
                y_test_pred_iter, errors_iter, mass_ejected_test_iter = i_shot_dummy(data_trimmed, cell_type_test=cell_type, i=iter, max_sample_sets=300, is_chain=False)

            y_test_pred_cell[iter]       = y_test_pred_iter
            errors_cell[iter]            = errors_iter
        
        y_test_pred[cell_type]       = y_test_pred_cell
        errors[cell_type]            = errors_cell

    with open(Path("results/dummy_y_test_pred.json"), "w") as f:
        json.dump(y_test_pred, f)
    with open(Path("results/dummy_errors.json"), "w") as f:
        json.dump(errors, f)

    # SVM training loop
    print('SVM training (RegressorChain)')
    y_test_pred = {}
    errors = {}
    
    cell_types = list(data_trimmed['Cell-Description'].unique())    
    # cell_types = ['KULR 18650-K330'] ### if you want to test quickly for just one cell
    for cell_type in cell_types:
        print(cell_type)
        
        y_test_pred_cell = {}
        errors_cell = {}

        max_i = data_trimmed.value_counts('Cell-Description')[cell_type]
        for iter in range(max_i): # range(max_i):
            print(iter)
            if iter == 0:
                y_test_pred_iter, errors_iter, mass_ejected_test_iter = zero_shot_svm(data_trimmed, cell_type_test=cell_type)
            else:
                y_test_pred_iter, errors_iter, mass_ejected_test_iter = i_shot_svm(data_trimmed, cell_type_test=cell_type, i=iter, max_sample_sets=300)

            y_test_pred_cell[iter]       = y_test_pred_iter
            errors_cell[iter]            = errors_iter
        
        y_test_pred[cell_type]       = y_test_pred_cell
        errors[cell_type]            = errors_cell

    with open(Path("results/svm_chain_y_test_pred.json"), "w") as f:
        json.dump(y_test_pred, f)
    with open(Path("results/svm_chain_errors.json"), "w") as f:
        json.dump(errors, f)

    # # SVM training loop
    # print('SVM training (MultiOutput)')
    # y_test_pred = {}
    # errors = {}
    
    # cell_types = list(data_trimmed['Cell-Description'].unique())    
    # # cell_types = ['KULR 18650-K330'] ### if you want to test quickly for just one cell
    # for cell_type in cell_types:
    #     print(cell_type)
        
    #     y_test_pred_cell = {}
    #     errors_cell = {}

    #     max_i = data_trimmed.value_counts('Cell-Description')[cell_type]
    #     for iter in range(max_i): # range(max_i):
    #         print(iter)
    #         if iter == 0:
    #             y_test_pred_iter, errors_iter, mass_ejected_test_iter = zero_shot_svm(data_trimmed, cell_type_test=cell_type, is_chain=False)
    #         else:
    #             y_test_pred_iter, errors_iter, mass_ejected_test_iter = i_shot_svm(data_trimmed, cell_type_test=cell_type, i=iter, max_sample_sets=300, is_chain=False)

    #         y_test_pred_cell[iter]       = y_test_pred_iter
    #         errors_cell[iter]            = errors_iter
        
    #     y_test_pred[cell_type]       = y_test_pred_cell
    #     errors[cell_type]            = errors_cell

    # with open(Path("results/svm_multioutput_y_test_pred.json"), "w") as f:
    #     json.dump(y_test_pred, f)
    # with open(Path("results/svm_multioutput_errors.json"), "w") as f:
    #     json.dump(errors, f)

    # # XGBoost training loop
    # print('XGBoost training')
    # y_test_pred = {}
    # errors = {}
    
    # cell_types = list(data_trimmed['Cell-Description'].unique())    
    # # cell_types = ['KULR 18650-K330'] ### if you want to test quickly for just one cell
    # for cell_type in cell_types:
    #     print(cell_type)
        
    #     y_test_pred_cell = {}
    #     errors_cell = {}

    #     max_i = data_trimmed.value_counts('Cell-Description')[cell_type]
    #     for iter in range(max_i): # range(max_i):
    #         print(iter)
    #         if iter == 0:
    #             y_test_pred_iter, errors_iter, mass_ejected_test_iter = zero_shot_xgb(data_trimmed, cell_type_test=cell_type)
    #         else:
    #             y_test_pred_iter, errors_iter, mass_ejected_test_iter = i_shot_xgb(data_trimmed, cell_type_test=cell_type, i=iter, max_sample_sets=300)

    #         y_test_pred_cell[iter]       = y_test_pred_iter
    #         errors_cell[iter]            = errors_iter
        
    #     y_test_pred[cell_type]       = y_test_pred_cell
    #     errors[cell_type]            = errors_cell

    # with open(Path("results/xgb_y_test_pred.json"), "w") as f:
    #     json.dump(y_test_pred, f)
    # with open(Path("results/xgb_errors.json"), "w") as f:
    #     json.dump(errors, f)