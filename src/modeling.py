import numpy as np
from src.data import Split
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from itertools import combinations
from xgboost import XGBRegressor
import GPUtil

# Features for dummy models
features_metadata = [
    'Cell-Description',
    ]
features_ejected_mass = [
    'Total-Mass-Ejected-g', # overall mass loss
    'Total Ejected Mass Fraction [g/g]', # overall mass loss
    'Unrecovered Mass Fraction [g/g]', # unrecovered mass
    'Body Mass Remaining Fraction [g/g]', # body mass loss
    'Positive Ejected Mass Fraction [g/g]', # positive end mass loss
    'Negative Ejected Mass Fraction [g/g]', # negative end mass loss
]
# Target variables
targets = [
    'Total Heat Output [kJ/A*h]',
    'Cell Body Heat Output [kJ/A*h]',
    'Positive Heat Output [kJ/A*h]',
    'Negative Heat Output [kJ/A*h]',
]

def scores(y_pred,y_test):
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    kl_divergence = entropy(y_pred, y_test)
    return mse, rmse, kl_divergence

def zero_shot_svm(data, cell_type_test, is_chain=True):
    if is_chain:
        svm = make_pipeline(StandardScaler(), RegressorChain(SVR(C=1, epsilon=0.1, kernel='linear'), order=[1,3,2,0]))
    else:
        svm = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=1, epsilon=0.1, kernel='linear')))
    y_test_pred, errors, mass_ejected_test = zero_shot(data, cell_type_test, svm)
    return y_test_pred, errors, mass_ejected_test

def zero_shot_xgb(data, cell_type_test, is_chain=True):
    if GPUtil.getAvailable():
        device = "cuda"
    else:
        device = "cpu"
    if is_chain:
        xgb = make_pipeline(StandardScaler(), RegressorChain(XGBRegressor(device=device), order=[1,3,2,0]))
    else:
        xgb = make_pipeline(StandardScaler(), MultiOutputRegressor(XGBRegressor(device=device)))
    y_test_pred, errors, mass_ejected_test = zero_shot(data, cell_type_test, xgb)
    return y_test_pred, errors, mass_ejected_test

def zero_shot_dummy(data, cell_type_test, is_chain=True):
    '''
    The zero-shot 'dummy' model is a linear regression on the ejected mass features, using
    all other cell types as training data and making predictions on the held-out cell type.
    '''
    if is_chain:
        dummy_regressor = make_pipeline(StandardScaler(), RegressorChain(LinearRegression(), order=[1,3,2,0]))
    else:
        dummy_regressor = make_pipeline(StandardScaler(), MultiOutputRegressor(LinearRegression()))
    # Remove all data columns except mass and heat and cell descriptions
    data = data[features_metadata + features_ejected_mass + targets]
    y_test_pred, errors, mass_ejected_test = zero_shot(data, cell_type_test, dummy_regressor, is_dummy=True)
    return y_test_pred, errors, mass_ejected_test

def zero_shot(data, cell_type_test, Regressor, is_dummy=False):
    # for zero shot case, one cell type is test set, all other cell types are training
    data_split = Split(data, 'cell_type_split', cell_type_test=cell_type_test)
    if not is_dummy:
        x_train, y_train, x_test, y_test = data_split.get_splits()
    else:
        # Get rid of one-hot encoded columns in feature matrices
        x_train, y_train, x_test, y_test = data_split.get_splits(is_numpy=False)
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        x_train = x_train[features_ejected_mass].copy().to_numpy()
        x_test = x_test[features_ejected_mass].copy().to_numpy()

    # Train model
    Regressor.fit(x_train, y_train)

    # Predictions
    # y_train_pred = Regressor.predict(x_train)
    y_test_pred = Regressor.predict(x_test)
    errors = calculate_errors(y_test, y_test_pred)

    # Collect test set masses: total (g), total (g/g), positive(g/g), negative(g/g)
    if not is_dummy:
        mass_ejected_test = np.vstack((x_test[:,3], x_test[:,4], x_test[:,5], x_test[:,6])).T
    else:
        mass_ejected_test = np.vstack((x_test[:,0], x_test[:,1], x_test[:,4], x_test[:,5])).T
    
    return y_test_pred.tolist(), errors, mass_ejected_test.tolist()

def i_shot_svm(data, cell_type_test, i, max_sample_sets=1000, is_chain=True):
    '''
    Use 'i' cells from the test cell type for training, training models for each possible combination
    of 'i' cells from 'N' cells of that type (N choose i). Return results for each combination.
    '''
    if is_chain:
        svm = make_pipeline(StandardScaler(), RegressorChain(SVR(C=1, epsilon=0.1, kernel='linear'), order=[1,3,2,0]))
    else:
        svm = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=1, epsilon=0.1, kernel='linear')))
    y_test_pred, errors, mass_ejected_test = i_shot(data, cell_type_test, i, svm, max_sample_sets)
    return y_test_pred, errors, mass_ejected_test

def i_shot_xgb(data, cell_type_test, i, max_sample_sets=1000, is_chain=True):
    '''
    Use 'i' cells from the test cell type for training, training models for each possible combination
    of 'i' cells from 'N' cells of that type (N choose i). Return results for each combination.
    '''
    if GPUtil.getAvailable():
        device = "cuda"
    else:
        device = "cpu"
    if is_chain:
        xgb = make_pipeline(StandardScaler(), RegressorChain(XGBRegressor(device=device), order=[1,3,2,0]))
    else:
        xgb = make_pipeline(StandardScaler(), MultiOutputRegressor(XGBRegressor(device=device)))
    y_test_pred, errors, mass_ejected_test = i_shot(data, cell_type_test, i, xgb, max_sample_sets)
    return y_test_pred, errors, mass_ejected_test

def i_shot_dummy(data, cell_type_test, i, max_sample_sets=1000, is_chain=True):
    '''
    Use 'i' cells from the test cell type for training, training models for each possible combination
    of 'i' cells from 'N' cells of that type (N choose i). Return results for each combination.
    The 'dummy' i-shot model is a linear regression on just the recorded samples from the cell type of interest,
    using only the ejected mass features, neglecting any learning from other cells or metadata
    '''
    if is_chain:
        dummy_regressor = make_pipeline(StandardScaler(), RegressorChain(LinearRegression(), order=[1,3,2,0]))
    else:
        dummy_regressor = make_pipeline(StandardScaler(), MultiOutputRegressor(LinearRegression()))
    # Remove all data columns except mass and heat and cell description
    data = data[features_metadata + features_ejected_mass + targets]
    y_test_pred, errors, mass_ejected_test = i_shot(data, cell_type_test, i, dummy_regressor, max_sample_sets, is_dummy=True)
    return y_test_pred, errors, mass_ejected_test

def i_shot(data, cell_type_test, i, Regressor, max_sample_sets=1000, is_dummy=False):
    '''
    Use 'i' cells from the test cell type for training, training models for each possible combination
    of 'i' cells from 'N' cells of that type (N choose i). Return results for each combination.
    '''
    data_split = Split(data, 'cell_type_split', cell_type_test=cell_type_test)
    if not is_dummy:
        x_train, y_train, x_test, y_test = data_split.get_splits()
    else:
        # Get rid of one-hot encoded columns in feature matrices
        x_train, y_train, x_test, y_test = data_split.get_splits(is_numpy=False)
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        x_train = x_train[features_ejected_mass].copy().to_numpy()
        x_test = x_test[features_ejected_mass].copy().to_numpy()

    sample_sets = list(combinations(range(len(x_test)), i))
    # Can result in a huge number of possible combinations. Only try a random selection if it's too much.
    if len(sample_sets) > max_sample_sets:
        rand = np.random.randint(low=0, high=len(sample_sets)-1, size=max_sample_sets)
        sample_sets = [sample_sets[i] for i in rand]

    # Instantiate output variables
    errors = []
    y_test_pred = []
    mass_ejected_test = []
    
    # Iterate through the sets of samples
    for samples in sample_sets:
        samples = list(samples)
        # make combo train: samples + all other cell types
        if not is_dummy:
            x_train_iter = np.vstack([x_train, np.array(x_test)[samples]])
            y_train_iter = np.vstack([y_train, np.array(y_test)[samples]])
        else:
            # Dummy model only uses samples from the held-out cell type
            x_train_iter = np.array(x_test)[samples]
            y_train_iter = np.array(y_test)[samples]

        # Train model
        Regressor.fit(x_train_iter, y_train_iter)
        
        # Predictions
        y_test_pred_iter = Regressor.predict(x_test)
        errors_iter = calculate_errors(y_test, y_test_pred_iter)

        # Collect test set masses: total (g), total (g/g), positive(g/g), negative(g/g)
        if not is_dummy:
            mass_ejected_test_iter = np.vstack((x_test[:,3], x_test[:,4], x_test[:,5], x_test[:,6])).T
        else:
            mass_ejected_test_iter = np.vstack((x_test[:,0], x_test[:,1], x_test[:,4], x_test[:,5])).T

        # Append outputs
        y_test_pred.append(y_test_pred_iter.tolist())
        errors.append(errors_iter)
        mass_ejected_test.append(mass_ejected_test_iter.tolist())

    return y_test_pred, errors, mass_ejected_test

def calculate_errors(y, y_pred):
    mse_total, rmse_total, kl_total = scores(y_pred[:,0],y[:,0])
    mse_body, rmse_body, kl_body = scores(y_pred[:,1],y[:,1])
    mse_pos, rmse_pos, kl_pos = scores(y_pred[:,2],y[:,2])
    mse_neg, rmse_neg, kl_neg = scores(y_pred[:,3],y[:,3])
    
    errors = {
        'total': {
            'mse': mse_total.tolist(),
            'rmse': rmse_total.tolist(),
            'kl': kl_total.tolist(),
        }, 
        'body': {
            'mse': mse_body.tolist(),
            'rmse': rmse_body.tolist(),
            'kl': kl_body.tolist(),
        },
        'positive': {
            'mse': mse_pos.tolist(),
            'rmse': rmse_pos.tolist(),
            'kl': kl_pos.tolist(),
        },
        'negative': {
            'mse': mse_neg.tolist(),
            'rmse': rmse_neg.tolist(),
            'kl': kl_neg.tolist(),
        }
    }
    return errors