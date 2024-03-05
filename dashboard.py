import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set(style='dark')
st.header('Bike Sharing Rental')
button_radio= st.radio('Pilih Keperluanmu!', options=('Monitoring', 'Prediksi'))
if button_radio == 'Monitoring':
    def formatter(x, pos):
        return '{:,.0f}'.format(x/1)

    main_bike_df = pd.read_csv('main_bike_df.csv')
    datetime_columns = ["dteday"]
    main_bike_df.sort_values(by="dteday", inplace=True)
    main_bike_df.reset_index(inplace=True)
    for column in datetime_columns:
        main_bike_df[column] = pd.to_datetime(main_bike_df[column])

    min_date = main_bike_df['dteday'].min()
    max_date = main_bike_df['dteday'].max()

    with st.sidebar:
        st.header('Date Filter')
        
        start_date, end_date = st.date_input(
        label= 'Tanggal',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
        )
        st.image('https://as2.ftcdn.net/v2/jpg/03/52/62/49/1000_F_352624956_Kdv5REmEuvxgUbY9yv83wYJc8Vh0rrj1.jpg')
    main_bike_edit_df = main_bike_df[(main_bike_df["dteday"] >= str(start_date)) & 
                        (main_bike_df["dteday"] <= str(end_date))]


    st.subheader('Bike Sharing Rental Performance Dashboard')

    tab1, tab2, tab3 = st.tabs(["Overall Performance", "Days Performance", "Seasons Performance"])

    with tab1:

        
        monthly_data = main_bike_edit_df.resample('M', on='dteday').sum()
        plt.figure(figsize=(10, 5))

        sns.lineplot(
            x=monthly_data.index,
            y=monthly_data['cnt'],
            marker='o',
            linewidth=2,
            color="#72BCD4"
        )

        plt.title("Trend Jumlah Pengguna Bike Sharing Rental", loc="center", fontsize=20)
        plt.xlabel(None)
        plt.ylabel('Jumlah (orang)')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()
        st.pyplot()
        with st.expander("See explanation"):
            st.write(
            """Chart diatas merupakan performa dari Bike Sharing Rental secara keseluruhan. 
            Anda bisa mengatur range tanggal pada sidebar disamping.
            """
            )


    with tab2:

        plt.figure(figsize=(10, 5))
        df_workingDay = main_bike_edit_df.groupby(by='workingday').cnt.sum().reset_index()
        # df_workingDay = days_perform_df.reset_index()
        ax = sns.barplot(

            x= 'workingday',
            y='cnt',
            data=df_workingDay
        )
        for p in ax.patches:
            ax.annotate(format(p.get_height(), ','),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 5),  
                    textcoords='offset points',
                    fontsize=10,
                    color='black')

        plt.gca().yaxis.set_major_formatter(FuncFormatter(formatter))
        # plt.gca().yaxis.set_major_locator(MultipleLocator(250000))
        plt.title("Jumlah Pengguna berdasarkan Jenis Hari", loc="center", fontsize=15)
        plt.xticks(ticks=[0, 1], labels=['Holidays or Weekends', 'Weekdays'])
        plt.ylabel('Jumlah Pengguna (Orang)')
        plt.xlabel(None)
        plt.tick_params(axis='x', labelsize=12)
        plt.show()
        st.pyplot()
        with st.expander("See explanation"):
            st.write(
            """Chart diatas merupakan performa dari Bike Sharing Rental berdasarkan tipe hari. Tipe 
            hari diatas ada 2 yaitu saat weekdays atau saat holidays/weekends. 
            Anda bisa mengatur range tanggal pada sidebar disamping.
            """
            )

    with tab3:
        df_bySeason= main_bike_edit_df.groupby(by='season').cnt.sum().reset_index()

        ax = sns.barplot(

            x= 'season',
            y='cnt',
            data= df_bySeason
        )
        for p in ax.patches:
            ax.annotate(format(p.get_height(), ','),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 5),  # Distance from text to bars
                    textcoords='offset points',
                    fontsize=10,
                    color='black')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(formatter))
        # plt.gca().yaxis.set_major_locator(MultipleLocator(150000))
        plt.title("Jumlah Pengguna Berdasarkan Musim", loc="center", fontsize=15)
        plt.xticks(ticks=[0, 1, 2, 3], labels=['Semi', 'Panas', 'Gugur', 'Dingin'])
        plt.ylabel('Jumlah Pengguna (Orang)')
        plt.xlabel(None)

        plt.show()
        st.pyplot()
        with st.expander("See explanation"):
            st.write(
            """Chart diatas merupakan performa dari Bike Sharing Rental berdasarkan tipe musim.
            Musim terdiri dari 4 musim yaitu musim semi, gugur, panas, dan dingin. 
            Anda bisa mengatur range tanggal pada sidebar disamping.
            """
            )
else:
    st.subheader('Bike Sharing Rental Prediction Tool')
    

    main_bike_df = pd.read_csv('main_bike_df.csv')
    datetime_columns = ["dteday"]
    main_bike_df.sort_values(by="dteday", inplace=True)
    main_bike_df.reset_index(inplace=True)
    for column in datetime_columns:
        main_bike_df[column] = pd.to_datetime(main_bike_df[column])
    monthly_data = main_bike_df.resample('M', on='dteday').sum()

    exponentialSmoothing_C2 = ExponentialSmoothing(monthly_data['cnt'], trend='multiplicative', damped_trend=False, seasonal='additive')
    hasilES = exponentialSmoothing_C2.fit()
    start_periode_forecast = monthly_data.index.max() + pd.DateOffset(months=1)
    end_periode_forecast = start_periode_forecast + pd.DateOffset(months=24)

    forecast_ESc2 = hasilES.predict(start= start_periode_forecast, end= end_periode_forecast)

    forecast_ESc2_df = pd.DataFrame({'Date': pd.date_range(start = '2012-12-31', periods=25, freq='M'),
                                'Jumlah Pengguna': forecast_ESc2})

    min_date = main_bike_df['dteday'].min()
    max_date = forecast_ESc2_df['Date'].max()
    with st.sidebar:
        st.header('Date Filter')
        
        start_date, end_date = st.date_input(
        label= 'Tanggal',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
        )
        if start_date.year > 2012:
            st.warning("Tanggal minimal tidak boleh setelah tahun 2012. Mohon pilih tanggal lain.")
            start_date = min_date 
        st.image('https://as2.ftcdn.net/v2/jpg/03/52/62/49/1000_F_352624956_Kdv5REmEuvxgUbY9yv83wYJc8Vh0rrj1.jpg')
        
        
    main_bike_edit_df = main_bike_df[(main_bike_df["dteday"] >= str(start_date)) & 
                            (main_bike_df["dteday"] <= str(end_date))]
    forecast_ESc2_edit_df = forecast_ESc2_df[(forecast_ESc2_df['Date'] >= str(start_date)) & 
                                (forecast_ESc2_df['Date'] <= str(end_date))]
    monthly_data = main_bike_edit_df.resample('M', on='dteday').sum().reset_index()    
    plt.figure(figsize=(20, 15))

    # Plot data aktual
    plt.plot(monthly_data['dteday'], monthly_data['cnt'], label='Actual', color='blue', linewidth = 2,)

    # Plot hasil forecast
    plt.plot(forecast_ESc2_edit_df['Date'], forecast_ESc2_edit_df['Jumlah Pengguna'], label='Forecast', color='purple', linewidth = 5)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(None)
    plt.ylabel('Jumlah Pengguna (Orang)',fontsize=20)
    plt.title('Prediksi Jumlah Pengguna untuk 2 Tahun Kedepan',fontsize=25)
    plt.legend(fontsize = 20)

    st.pyplot()
    with st.expander("See explanation"):
            st.write(
            """Chart diatas merupakan prediksi jumlah pengguna untuk 2 tahun kedepan. 
            Silahkan gunakan sidebar untuk mengatur tanggal :)
            """
            )
    
        
