# Graph Edge Score Formula

The edge score between two tokens i and j is computed as:

```
edge_score(i, j) = mean(|corr(f_i^k, f_j^k)|) for all common feature dimensions k
```

where f_i^k denotes the k-th feature of token i, and corr denotes the Pearson correlation coefficient.

Features used for edge scoring:
1. 1-day returns
2. 3-day rolling mean returns
3. 7-day rolling mean returns
4. 7-day rolling volatility
5. 14-day rolling volatility
6. Log-transformed volume
7. 7-day volume moving average
8. Twitter sentiment 7-day mean
9. Reddit sentiment 7-day mean
10. Telegram sentiment 7-day mean
11. Log price level
12. Drawdown from peak
13. Direct price correlation

All features are computed using only past/current observations (causal).
Edges are selected using quantile-based thresholding (top 15% of pairwise similarities).