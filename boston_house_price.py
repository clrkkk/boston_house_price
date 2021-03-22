
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

st.set_option('deprecation.showPyplotGlobalUse', False)

st.header("""
**Boston House Price Prediction App**       
""")
st.write("""
This app predicts the **Boston House Price**!\n
reference: https://github.com/dataprofessor/streamlit_freecodecamp/blob/main/app_9_regression_boston_housing/boston-house-ml-app.py
""")
st.write('---')


# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
    RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
    AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
    DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
    RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
    TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
    PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
    LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Make training set and test set; Build Regression Model
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.1)
model = RandomForestRegressor(n_estimators = 50, criterion = "mse",max_depth =5, bootstrap = True, n_jobs = 4 ) 
model.fit(X_train, y_train)
predict_test = model.predict(X_test)

# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV based on RF')
st.write(prediction)

st.header('MSE of Prediction')
st.write( mean_squared_error(predict_test, y_test))
st.write('---')
# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
sh = shap.summary_plot(shap_values, X)
st.pyplot(sh)
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
st.write('---')

#Heat map 
st.header('Correlation Heat Map')
corrmat = X.corr()
plt.subplots(figsize=(13,13))
sns.heatmap (corrmat, vmax=0.8, square = True)
st.pyplot(plt)
st.write('---')

#neural network prediction
st.header('ANN Prediction')

#normalization
X_ann_train=(X_train - X_train.mean())/X_train.std()
df_scaled = (df-X_train.mean())/X_train.std()
X_ann=(X - X.mean())/X.std()

#ANN model
MLP = MLPRegressor(hidden_layer_sizes = (50,50,50), activation = 'relu', alpha = 0.0001, max_iter = 5000)
MLP.fit(X_ann_train, y_train)
MLP_predict = MLP.predict(df_scaled)
st.write(MLP_predict)

#rmse of ANN model

st.header('MSE & RMSE of ANN Prediction')
X_ann_test=(X_test - X_test.mean())/X_test.std()
MLP_predict_test = MLP.predict(X_ann_test)
mse = mean_squared_error(MLP_predict_test, y_test)
rmse = sqrt(mse)
st.write(mse, rmse)
