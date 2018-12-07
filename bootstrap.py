import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


size = 10
gradient = 2
intercept = 100
noise_scale = 5

df = pd.DataFrame({'x': range(size)})
df['y'] = round(df.x*gradient + intercept + noise_scale*np.random.normal(size=size))

ax = df.plot(x='x')
ax.set_ylim(0)



def bootstrap_simple(df, split_idx, n):
    results = pd.DataFrame(index=df[split_idx:].index)

    for i in range(n):
        df_train = df[:split_idx]
        df_test = df[split_idx:]
        df_train = df_train.sample(frac=1, replace=True)
        model = np.poly1d(np.polyfit(df_train.x, df_train.y, deg=1))
        y_hat = model(df_test.x)
        results[i] = y_hat
    return results

def calc_intervals(result, title='intervals', plot=True):
    stats = pd.DataFrame({'mean': result.mean(axis=1),
                          'lower': result.quantile(q=0.05, axis=1),
                          'upper': result.quantile(q=0.95, axis=1)},
                        index=result.index)
    if plot:
        ax = stats.plot(title=title)
        ax.set_ylim(0)
        return ax
    else:
        return stats

def flatten_to_records(df):
    result = pd.Series(name='x')
    for i in df.x.unique():
        y = int(df[df.x == i].y.values[0])
        result = result.append(pd.Series([i]*y), ignore_index=True)
    return result



def bootstrap_records(df, split_idx, n):
    results = pd.DataFrame(index=df[split_idx:].index)
    df_train = df[:split_idx]
    df_test = df[split_idx:]
    s = flatten_to_records(df_train)

    for i in range(n):
        s_sample = s.sample(frac=1, replace=True)
        s_sample = s_sample.value_counts().sort_index()
        model = np.poly1d(np.polyfit(s_sample.index, s_sample.tolist(), deg=1))
        y_hat = model(df_test.x)
        results[i] = y_hat
    return results



def bootstrap_stochastic(df, split_idx, n):
    results = pd.DataFrame(index=df[split_idx:].index)
    df_train = df[:split_idx]
    df_test = df[split_idx:]
    s = flatten_to_records(df_train)

    for i in range(n):
        s_sample = s.sample(frac=1, replace=True)
        s_sample = s_sample.value_counts().sort_index()
        model = np.poly1d(np.polyfit(s_sample.index, s_sample.tolist(), deg=1))
        y_hat = model(df_test.x)
        vals = []
        for v in y_hat:
            vals.append(np.random.poisson(v))
        results[i] = vals
    return results



