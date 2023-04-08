# GEE_Sparse

![Preview Image](https://github.com/xihan-qin/GEE_Sparse/blob/main/GEE_vs_GEE_sparse.png)


<table>
<thead>
  <tr>
    <th rowspan="2">Data Set</th>
    <th colspan="2">Lap = T, Diag = T, Cor = T</th>
    <th colspan="2">Lap = T, Diag = T, Cor = F</th>
    <th colspan="2">Lap = T, Diag = F, Cor = T</th>
    <th colspan="2">Lap = T, Diag = F, Cor = F</th>
  </tr>
  <tr>
    <th>GEE</th>
    <th>GEE_sparse</th>
    <th>GEE</th>
    <th>GEE_sparse</th>
    <th>GEE</th>
    <th>GEE_sparse</th>
    <th>GEE</th>
    <th>GEE_sparse</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CiteSeer</td>
    <td>0.096</td>
    <td><strong>0.040</strong></td>
    <td>0.097</td>
    <td><strong>0.032</strong></td>    
    <td>0.053</td>
    <td><strong>0.032</strong></td>    
    <td>0.051</td>
    <td><strong>0.030</strong></td>        
  </tr>
  <tr>
    <td>Cora</td>
    <td>0.083</td>
    <td><strong>0.028</strong></td>    
    <td>0.088</td>
    <td><strong>0.027</strong></td>        
    <td>0.068</td>
    <td><strong>0.033</strong></td>            
    <td>0.068</td>
    <td><strong>0.026</strong></td>              
  </tr>
  <tr>
    <td>Proteins-all</td>
    <td>2.268</td>
    <td><strong>0.438</strong></td>      
    <td>2.259</td>
    <td><strong>0.419</strong></td>          
    <td>1.866</td>
    <td><strong>0.391</strong></td>          
    <td>1.846</td>
    <td><strong>0.478</strong></td>      
  </tr>
  <tr>
    <td>PubMed</td>
    <td>0.776</td>
    <td><strong>0.194</strong></td>       
    <td>0.739</td>
    <td><strong>0.201</strong></td>       
    <td>0.673</td>
    <td><strong>0.208</strong></td>       
    <td>0.560</td>
    <td><strong>0.199</strong></td>   
  </tr>
  <tr>
    <td>CL-100K-1d8-L9</td>
    <td>4.926</td>
    <td>1.091</td>
    <td>4.850</td>
    <td>0.978</td>
    <td>4.166</td>
    <td>0.992</td>
    <td>3.823</td>
    <td>1.095</td>
  </tr>
  <tr>
    <td>CL-100K-1d8-L5</td>
    <td>604.018</td>
    <td>174.552</td>
    <td>585.288</td>
    <td>147.790</td>
    <td>633.746</td>
    <td>118.705</td>
    <td>571.360</td>
    <td>123.691</td>
  </tr>
</tbody>
</table>

|             | Lap = T, Diag = T, Cor = T   || Lap = T, Diag = T, Cor = F   ||
Data Set      |     GEE      |   GEE_sparse   |     GEE      |   GEE_sparse   |
  ----------- | :----------: | -------------: | :----------: |  -------------:|
| CiteSeer    |     0.096    |      0.040     |     0.097    |      0.032     |



