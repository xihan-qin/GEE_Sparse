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
    <th>GEE(s)</th>
    <th>GEE_sparse(s)</th>
    <th>GEE(s)</th>
    <th>GEE_sparse(s)</th>
    <th>GEE(s)</th>
    <th>GEE_sparse(s)</th>
    <th>GEE(s)</th>
    <th>GEE_sparse(s)</th>
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
    <td><strong>1.091</strong></td>   
    <td>4.850</td>
    <td><strong>0.978</strong></td>   
    <td>4.166</td>
    <td><strong>0.992</strong></td>   
    <td>3.823</td>
    <td><strong>1.095</strong></td>   
  </tr>
  <tr>
    <td>CL-100K-1d8-L5</td>
    <td>604.018</td>
    <td><strong>174.552</strong></td>   
    <td>585.288</td>
    <td><strong>147.790</strong></td>   
    <td>633.746</td>
    <td><strong>118.705</strong></td>   
    <td>571.360</td>
    <td><strong>123.691</strong></td>   
  </tr>
</tbody>
</table>



