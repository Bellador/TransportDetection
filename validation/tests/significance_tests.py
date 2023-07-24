import pandas as pd
from yachalk import chalk
from scipy.stats import mannwhitneyu


def weather_test():

    FILTERED_CSV_STATISTICS_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\location_statistics_45s_buffer_ORIGINAL_FINISH_FILTERED.csv"

    df = pd.read_csv(FILTERED_CSV_STATISTICS_PATH, delimiter=';')

    # Test for significant difference of good vs bad weather effects on active, motorised and public transport
    transport_modes = ["active", "motorised", "public"]

    for mode in transport_modes:
        print("--"*30)
        print(f'Testing weather significance: {mode} ')
        bad_weather_df = df.loc[df["badWeather"] == 1, [mode]]
        good_weather_df = df.loc[df["badWeather"] == 0, [mode]]
        # Perform the Mann-Whitney U test
        stat, p = mannwhitneyu(bad_weather_df, good_weather_df)
        # Print the results
        print('Statistic:', stat)
        print('p-value:', p)
        # Interpret the results
        alpha = 0.05
        if p > alpha:
            print('[*] Same distribution (fail to reject H0)')
        else:
            print(chalk.bold.red.bg_black('[!] Different distribution (reject H0)'))

        print("--" * 30)

def weather_pedest_cyclist_test():

    FILTERED_CSV_STATISTICS_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\from_cluster\location_statistics_45s_buffer_ORIGINAL_FINISH_FILTERED.csv"

    df = pd.read_csv(FILTERED_CSV_STATISTICS_PATH, delimiter=';')

    # Test for significant difference of good vs bad weather effects on active, motorised and public transport
    transport_modes = ["pedestrian", "cyclist", "motorised"]

    for mode in transport_modes:
        print("--"*30)
        print(f'Testing weather significance: {mode} ')
        bad_weather_df = df.loc[df["badWeather"] == 1, [mode]]
        good_weather_df = df.loc[df["badWeather"] == 0, [mode]]
        # Perform the Mann-Whitney U test
        stat, p = mannwhitneyu(bad_weather_df, good_weather_df)
        # Print the results
        print('Statistic:', stat)
        print('p-value:', p)
        # Interpret the results
        alpha = 0.05
        if p > alpha:
            print('[*] Same distribution (fail to reject H0)')
        else:
            print(chalk.bold.red.bg_black('[!] Different distribution (reject H0)'))

        print("--" * 30)

def covid_quartier_test():
    '''
    test for statistical significance between the pre and post covid timerperiod of the Top 5 quartiers for the transport modes:
    (1) pedestrian, (2) cyclist, (3) motorised transport
    Loaded data was returned from boxplots_COVID_by_quartier() in output_plotting.py
    :return:
    '''

    data_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\verification\tests\COVID_quartier_data.csv"

    df = pd.read_csv(data_PATH, delimiter=';', encoding='utf-8')

    # Test for significant difference of good vs bad weather effects on active, motorised and public transport
    transport_modes = ["pedestrians", "cyclists", "motorised"]
    # Top 5 quartiers to test significant difference between pre and post covid for
    quartiers = ['Sorbonne', 'Clignancourt', 'Halles', "Saint-Germain-l'Auxerrois", 'Saint-Germain-des-Prés']

    for quartier in quartiers:
        for mode in transport_modes:
            print("--"*30)
            print(f'Testing pre/post covid significance:\nQuartier: {quartier}\nMode: {mode}')
            pre_covid_df = df.loc[(df["time"] == "pre_covid") & (df["quartier"] == quartier) & (df["mode"] == mode), ["value"]]
            post_covid_df = df.loc[(df["time"] == "post_covid") & (df["quartier"] == quartier) & (df["mode"] == mode), ["value"]]
            # Perform the Mann-Whitney U test
            stat, p = mannwhitneyu(pre_covid_df, post_covid_df)
            # Print the results
            print('Statistic:', stat)
            print('p-value:', p)
            # Interpret the results
            alpha = 0.05
            if p > alpha:
                print('[*] Same distribution (fail to reject H0)')
            else:
                print(chalk.bold.red.bg_black('[!] Different distribution (reject H0)'))

            print("--" * 30)

# covid_quartier_test()
weather_pedest_cyclist_test()