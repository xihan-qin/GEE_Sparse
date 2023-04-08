# GEE_Sparse

![Preview Image](https://github.com/xihan-qin/GEE_Sparse/blob/main/GEE_vs_GEE_sparse.png)


<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-fymr{border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">Data Set</th>
    <th class="tg-0pky" colspan="2">Lap = T, Diag = T, Cor = T</th>
    <th class="tg-0pky" colspan="2">Lap = T, Diag = T, Cor = F</th>
    <th class="tg-0pky" colspan="2">Lap = T, Diag = F, Cor = T</th>
    <th class="tg-0pky" colspan="2">Lap = T, Diag = F, Cor = F</th>
  </tr>
  <tr>
    <th class="tg-0pky">GEE</th>
    <th class="tg-0pky">GEE_sparse</th>
    <th class="tg-0pky">GEE</th>
    <th class="tg-0pky">GEE_sparse</th>
    <th class="tg-0pky">GEE</th>
    <th class="tg-0pky">GEE_sparse</th>
    <th class="tg-0pky">GEE</th>
    <th class="tg-0pky">GEE_sparse</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">CiteSeer</td>
    <td class="tg-0pky">0.096</td>
    <td class="tg-fymr">0.040</td>
    <td class="tg-0pky">0.097</td>
    <td class="tg-fymr">0.032</td>
    <td class="tg-0pky">0.053</td>
    <td class="tg-fymr">0.032</td>
    <td class="tg-0pky">0.051</td>
    <td class="tg-fymr">0.030</td>
  </tr>
  <tr>
    <td class="tg-0pky">Cora</td>
    <td class="tg-0pky">0.083</td>
    <td class="tg-0pky">0.028</td>
    <td class="tg-0pky">0.088</td>
    <td class="tg-0pky">0.027</td>
    <td class="tg-0pky">0.068</td>
    <td class="tg-0pky">0.033</td>
    <td class="tg-0pky">0.068</td>
    <td class="tg-0pky">0.026</td>
  </tr>
  <tr>
    <td class="tg-0pky">Proteins-all</td>
    <td class="tg-0pky">2.268</td>
    <td class="tg-fymr">0.438</td>
    <td class="tg-0pky">2.259</td>
    <td class="tg-fymr">0.419</td>
    <td class="tg-0pky">1.866</td>
    <td class="tg-fymr">0.391</td>
    <td class="tg-0pky">1.846</td>
    <td class="tg-fymr">0.478</td>
  </tr>
  <tr>
    <td class="tg-0pky">PubMed</td>
    <td class="tg-0pky">0.776</td>
    <td class="tg-fymr">0.194</td>
    <td class="tg-0pky">0.739</td>
    <td class="tg-fymr">0.201</td>
    <td class="tg-0pky">0.673</td>
    <td class="tg-fymr">0.208</td>
    <td class="tg-0pky">0.560</td>
    <td class="tg-fymr">0.199</td>
  </tr>
  <tr>
    <td class="tg-0pky">CL-100K-1d8-L9</td>
    <td class="tg-0pky">4.926</td>
    <td class="tg-fymr">1.091</td>
    <td class="tg-0pky">4.850</td>
    <td class="tg-fymr">0.978</td>
    <td class="tg-0pky">4.166</td>
    <td class="tg-fymr">0.992</td>
    <td class="tg-0pky">3.823</td>
    <td class="tg-fymr">1.095</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">CL-100K-1d8-L5</span></td>
    <td class="tg-0pky">604.018</td>
    <td class="tg-fymr">174.552</td>
    <td class="tg-0pky">585.288</td>
    <td class="tg-fymr">147.790</td>
    <td class="tg-0pky">633.746</td>
    <td class="tg-fymr">118.705</td>
    <td class="tg-0pky">571.360</td>
    <td class="tg-fymr">123.691</td>
  </tr>
</tbody>
</table>

|             | Lap = T, Diag = T, Cor = T   || Lap = T, Diag = T, Cor = F   ||
Data Set      |     GEE      |   GEE_sparse   |     GEE      |   GEE_sparse   |
  ----------- | :----------: | -------------: | :----------: |  -------------:|
| CiteSeer    |     0.096    |      0.040     |     0.097    |      0.032     |



