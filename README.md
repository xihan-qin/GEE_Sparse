# GEE_Sparse

![Preview Image](https://github.com/xihan-qin/GEE_Sparse/blob/main/GEE_vs_GEE_sparse.png)


|              Stage | Direct Products | ATP Yields |
| -----------------: | --------------: | ---------: |
|         Glycolysis |          2 ATP              ||
| ^^                 |          2 NADH |   3--5 ATP |
| Pyruvaye oxidation |          2 NADH |      5 ATP |
|  Citric acid cycle |          2 ATP              ||
| ^^                 |          6 NADH |     15 ATP |
| ^^                 |          2 FADH |      3 ATP |
|                               30--32 ATP        |||


|             | Lap = T, Diag = T, Cor = T   || Lap = T, Diag = T, Cor = F   ||
Data Set      |     GEE      |   GEE_sparse   |     GEE      |   GEE_sparse   |
  ----------- | :----------: | -------------: | :----------: |  -------------:|
| CiteSeer    |     0.096    |      0.040     |     0.097    |      0.032     |



