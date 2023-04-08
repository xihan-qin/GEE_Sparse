# GEE_Sparse

![Preview Image](https://github.com/xihan-qin/GEE_Sparse/blob/main/GEE_vs_GEE_sparse.png)


<table>
    <thead>
        <tr>
            <th>Layer 1</th>
            <th>Layer 2</th>
            <th>Layer 3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>L1 Name</td>
            <td rowspan=2>L2 Name A</td>
            <td>L3 Name A</td>
        </tr>
        <tr>
            <td>L3 Name B</td>
        </tr>
        <tr>
            <td rowspan=2>L2 Name B</td>
            <td>L3 Name C</td>
        </tr>
        <tr>
            <td>L3 Name D</td>
        </tr>
    </tbody>
</table>

|             | Lap = T, Diag = T, Cor = T   || Lap = T, Diag = T, Cor = F   ||
Data Set      |     GEE      |   GEE_sparse   |     GEE      |   GEE_sparse   |
  ----------- | :----------: | -------------: | :----------: |  -------------:|
| CiteSeer    |     0.096    |      0.040     |     0.097    |      0.032     |



