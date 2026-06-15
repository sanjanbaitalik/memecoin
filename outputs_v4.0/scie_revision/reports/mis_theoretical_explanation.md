# Greedy Maximal Independent Set Explanation

PRISM represents tokens as nodes in a graph and connects two nodes when their observed trajectories, sentiment dynamics, volatility profiles, or liquidity/risk characteristics are sufficiently similar.

An independent set is a subset of nodes with no direct high-redundancy edge between any pair of selected nodes. A maximal independent set is one that cannot be enlarged without violating that property.

Because the implementation uses a greedy procedure, the correct name is greedy maximal independent set rather than maximum independent set.

In this setting, the selected subset is diversified under the graph definition: it suppresses direct similarity edges and therefore reduces redundant token exposure before downstream sequence modeling.