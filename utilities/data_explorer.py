import pandas as pd
from itertools import combinations
from functools import reduce

from sweep_futures import SWEEP_PARAMETERS

#suffix_list.append(f"_{sweep_param}{val}")

def make_comparison_df(df, sweep_param, comp_metrics):
    grouped = df.groupby(sweep_param)
    param_names = [key for key in SWEEP_PARAMETERS.keys() if key != sweep_param]
    dfg_list = []
    for ind, val in enumerate(SWEEP_PARAMETERS[sweep_param]):
        dfg = grouped.get_group(val)
        metric_names = [metric+'_'+sweep_param+f'{val}' for metric in comp_metrics]
        dfg = dfg.rename(columns=dict(zip(comp_metrics,metric_names)))
        keyset = param_names + metric_names
        dfg = dfg[keyset]
        dfg_list.append(dfg)
    return reduce(lambda left, right: pd.merge(left, right, on=param_names, how='outer'), dfg_list)

def comparison_summary(df, sweep_param, comp_metrics):
    count_total = len(df)
    print(sweep_param)
    for combo in combinations(SWEEP_PARAMETERS[sweep_param],2):
        print(combo)
        for metric in comp_metrics:
            print(metric)
            col1 = metric+'_'+sweep_param+f'{combo[0]}'
            col2 = metric+'_'+sweep_param+f'{combo[1]}'
            count_equal = (df[col1] == df[col2]).sum()
            count_NaN1 = df[col1].isna().sum()
            count_NaN2 = df[col2].isna().sum()
            count_greater = ((df[col1] > df[col2]) | (~df[col1].isna() & df[col2].isna())).sum()
            count_less    = ((df[col1] < df[col2]) | (~df[col2].isna() & df[col1].isna())).sum()
            print('For '+metric+f', {count_equal}/{count_total} cases are equal between '+col1+' and '+col2)
            print('For '+metric+', '+col1+' was NaN ' f'{count_NaN1}/{count_total} times versus '
                    +f'{count_NaN2}/{count_total} times for '+col2)
            print('For '+metric+', '+col1+' was greater than '
                  +col2+ f' {count_greater}/{count_total} times (or not NaN when '
                  +col2+' was NaN)')
            print('For '+metric+', '+col2+' was greater than '
                  +col1+ f' {count_less}/{count_total} times (or not NaN when '
                  +col1+' was NaN)')

def main():
    df = pd.read_csv('sweep_results_post.csv')
    #df = dfbig.head(30)
    #print(df)
    sweep_param = "DEL"
    comp_metrics = ["final_dt", "Ni_fraction"]
    dfg = make_comparison_df(df, sweep_param, comp_metrics)
    comparison_summary(dfg, sweep_param, comp_metrics)

if __name__ == "__main__":
    main()

#with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.max_colwidth', None,
#                       'display.width', 1000):
#    print(final_df)