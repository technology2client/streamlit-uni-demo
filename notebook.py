import streamlit as st
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Functions
PROPHET = 'Prophet'
XGBOOST = 'XGBoost'

if __name__ == '__main__':
    def f_xgboost(input, date_column: str, objective_column: str):
        data = pd.DataFrame(input)

        X, y = np.vstack(data.index.values), np.vstack(data[objective_column].values)

        data_dmatrix = xgb.DMatrix(data=X, label=y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=15, alpha=10,
                                  n_estimators=1000)

        xg_reg.fit(X_train, y_train)

        preds = xg_reg.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))


        df_prediction = pd.DataFrame({
            date_column: input.loc[X_test.reshape(len(X_test)), date_column],
            'Actual_temp': y_test.reshape(len(y_test)),
            'Pred_temp': preds
        })
        st.write("RMSE: %f" % rmse)

        params = {"objective": "reg:squarederror", 'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': 15, 'alpha': 10}

        cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                            num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True)
        return cv_results, df_prediction
        # return None


    def f_prophet(input, date_column: str, objective_column: str):
        data = pd.DataFrame({
            'ds': input[date_column].values,
            'y': input[objective_column].values
        })
        n_split = round(len(data)*0.8)

        train_df = data[:n_split]
        test_df = data[n_split:]

        p_model = Prophet()
        p_model.fit(train_df)

        prediction = p_model.predict(test_df[['ds']])
        prediction = prediction[['ds', 'yhat']]

        df_prediction = pd.DataFrame({
            date_column: test_df['ds'].values,
            'Actual_temp': test_df['y'].values,
            'Pred_temp': prediction['yhat'].values
        })
        cv_results = None
        return cv_results, df_prediction

    @st.cache
    def convert_df(df):

        return df.to_csv().encode('utf-8')


    # Web structure


    st.markdown(
        "<img alt='FCC' src='https://profesoresim2019.upc.edu/wp/wp-content/uploads/2018/09/logo_UPC.png' width='57px' height='57px' style='text-align: center; float: left'></img>"
        "<img alt='T2C' src='data:image/png;base64,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' width='185px' height='57px' style='text-align: center; float: right'></img>"
        "<style>.block-container{ max-width: 55rem;}</style>", unsafe_allow_html=True)

    with st.sidebar:
        # Can be used wherever a "file-like" object is accepted:
        uploaded_file = st.file_uploader("Carga un nuevo CSV", type=["csv","xlsx"])
        file_separator = st.radio('Which is the CSV column separator?', options=[';',','])
        df_uploaded_file = pd.DataFrame()
        if uploaded_file is not None:
            df_uploaded_file = pd.read_csv(uploaded_file, sep=file_separator)

    tab1, tab2 = st.tabs(["Resultados", "Mapa"])

    with tab1:
        st.subheader("Resultados")

        # SHOW UPLOAD
        if not df_uploaded_file.empty:
            st.write(df_uploaded_file)
            desired_columns = st.multiselect('Select desired columns', options=df_uploaded_file.columns)
            date_column = st.radio('Which is the date variable', options=desired_columns)
            objective_column = st.radio('Which is the objective variable', options=desired_columns)

        # BUTTON
        results = None
        model = st.radio('Which model to use?', options=[PROPHET, XGBOOST])
        if st.button("Execute"):
            df_to_predict = df_uploaded_file[desired_columns]
            if model == PROPHET:
                results, df_prediction = f_prophet(input=df_to_predict, date_column=date_column, objective_column=objective_column)
            if model == XGBOOST:
                results, df_prediction = f_xgboost(input=df_to_predict, date_column=date_column, objective_column=objective_column)
            df_prediction[date_column] = pd.to_datetime(df_prediction[date_column], format='%d/%m/%Y')
            st.line_chart(df_prediction, x=date_column)

        # SHOW RESULTS
        if results:
            # results = f_xgboost(input=df_uploaded_file)
            st.write(results)

        # DOWNLOAD RESULTS
        csv = convert_df(df_uploaded_file)
        st.download_button(
            label="Descarga los resultados",
            data=csv,
            file_name='resultados.csv',
            mime='text/csv',
        )

    with tab2:
        st.subheader("Mapa")
        with st.spinner('Cargando el mapa'):
            df_coordinates = pd.DataFrame(
                data={'lat': [41.38945007324219, 36.72016], 'lon': [2.1317803859710693, -4.42034]})
            st.map(df_coordinates)

    # http://localhost:8501/
