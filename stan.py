import pystan
import pandas as pd


def main():
    df = pd.read_csv('./data/extracted_data.csv')
    
    # 天気を0,1,2に変換
    weather = df['weather'].tolist()
    for i, w in enumerate(weather):
        if w == 'sunny':
            weather[i] = 0
        elif w == 'cloudy':
            weather[i] = 1
        else:
            weather[i] = 2

    data = {
        'N': len(df),
        'K': 4,  # Kはカテゴリ数(今回は0,1,2,3の4つ)
        'D': len(df.columns),
        'Y': list(df['target'] + 1),
        'is_workday': list(df['is_working']),
        'weather': weather,
        'temperature': list(df['temperature'] / 40),
        'air_pressure': list(df['air_pressure']/ 1020) ,
        'humidity': list(df['humidity']/ 100),
        'study_in_morning': list(df['study_in_morning']),
        'go_out': list(df['go_out']),
        'bedtime': list(df['bedtime'] / 30),
        'sleep_latency': list(df['sleep_latency'] / 5),
        'sleep_duration': list(df['sleep_duration'] / 10)
    }

    code = """
        data {
            int N;
            int K;
            int D;
            int<lower=1, upper=K> Y[N];
            int<lower=0, upper=1> is_workday[N];
            int<lower=0, upper=2> weather[N];
            real temperature[N];
            real air_pressure[N];
            real humidity[N];
            int<lower=0, upper=1> study_in_morning[N];
            int<lower=0, upper=1> go_out[N];
            real bedtime[N];
            real sleep_latency[N];
            real sleep_duration[N];
        }

        transformed data {
            vector[D] zeros;
            zeros = rep_vector(0, D);
        }

        parameters {
            matrix[D, K-1] b_raw;
        }

        transformed parameters {
            simplex[K] theta[N];
            matrix[D, K] b;
            b = append_col(zeros, b_raw);
            for (n in 1:N)
                theta[n] = softmax((b[1] + b[2]*is_workday[n] + b[3]*weather[n] + b[4]*temperature[n] + b[5]*air_pressure[n] + b[6]*humidity[n] + b[7]*study_in_morning[n] + b[8]*go_out[n] + b[9]*bedtime[n] + b[10]*sleep_latency[n] + b[11]*sleep_duration[n])');
        }

        model {
            for (n in 1:N)
                Y[n] ~ categorical(theta[n]);
        }
    """
    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=3)
    la = fit.extract(permuted=True)
    print(fit)


if __name__ == "__main__":
    main()