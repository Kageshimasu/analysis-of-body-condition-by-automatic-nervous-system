import pystan
import pandas as pd
import numpy as np 


def main():
    df = pd.read_csv('./data/extracted_data.csv')

    weather_mapping = {'sunny': 0, 'cloudy': 1, 'rainy': 2}
    df['weather'] = df['weather'].map(weather_mapping)

    target = list(df['target'] + 1)
    df = df.drop(['target'], axis=1)
    ones = np.ones(len(df.values))
    df['temperature'] = df['temperature'] / 40
    df['air_pressure'] = df['air_pressure'] / 1020
    df['humidity'] = df['humidity'] / 100
    df['bedtime'] = df['bedtime'] / 30
    df['sleep_latency'] = df['sleep_latency'] / 5
    df['sleep_duration'] = df['sleep_duration'] / 10
    X = np.concatenate((ones.reshape(len(ones), 1), df.values),axis=1)

    data = {
        'N': len(df),
        'D': X.shape[1],
        'K': 4,  # Kはカテゴリ数(今回は0,1,2,3の4つ)
        'Y': target,
        'X': X
    }

    # df = pd.read_csv('./data/data2.csv')
    # target = list(df['Y'])   
    # df = df.drop(['Y'], axis=1)
    # df['Age'] = df['Age'] / 100
    # df['Income'] = df['Income'] / 1000
    # ones = np.ones(len(df.values))
    # X = np.concatenate((ones.reshape(len(ones), 1), df.values),axis=1)
    # print(X)

    # data = {
    #     'N': len(df),
    #     'D': X.shape[1],
    #     'K': 6,
    #     'Y': target,
    #     'X': X
    # }

    code = """
        data {
            int N;
            int D;
            int K;
            matrix[N,D] X;
            int<lower=1, upper=K> Y[N];
        }

        transformed data {
            vector[D] Zeros;
            Zeros = rep_vector(0,D);
        }

        parameters {
            matrix[D,K-1] b_raw;
        }

        transformed parameters {
            matrix[D,K] b;
            matrix[N,K] mu;
            b = append_col(Zeros, b_raw);
            mu = X*b;
        }

        model {
            for (n in 1:N)
                Y[n] ~ categorical(softmax(mu[n,]'));
        }
    """

    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=3)
    la = fit.extract(permuted=True)
    print(fit)


if __name__ == "__main__":
    main()