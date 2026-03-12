import pandas as pd


def main():
    # load data
    df = pd.read_csv('reddit_mentions_2024.csv')

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
    top25.to_csv('top25_stocks.csv', index=False)
    bottom25.to_csv('bottom25_stocks.csv', index=False)


if __name__ == '__main__':
    main()
