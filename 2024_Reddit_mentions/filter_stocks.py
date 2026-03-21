import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # load data
    df = pd.read_csv(os.path.join(SCRIPT_DIR, '../2024 Reddit mentions/reddit_mentions_2024.csv'))

    # aggregate mentions per ticker
    agg = df.groupby('ticker', as_index=False)['mentions'].sum()

    # sort descending
    sorted_df = agg.sort_values('mentions', ascending=False)

    # get top 25 and bottom 25
    top25 = sorted_df.head(25)
    bottom25 = sorted_df.tail(25)

    # output results
    print("Top 25 mentioned stocks:")
    print(top25.to_string(index=False))
    print("\nBottom 25 mentioned stocks:")
    print(bottom25.to_string(index=False))

    # save to csv if desired
    top25.to_csv(os.path.join(SCRIPT_DIR, 'top25_stocks.csv'), index=False)
    bottom25.to_csv(os.path.join(SCRIPT_DIR, 'bottom25_stocks.csv'), index=False)


if __name__ == '__main__':
    main()
