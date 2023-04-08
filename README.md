# GEE_Sparse



# GEE vs GEE_Sparse
## Data sets
Simulated datasets are generated using Stochastic Block Model (SBM).

The settings:
* 3 classes with class probabilities [0.2, 0.3, 0.5] 
* between class probability 0.1 and within class probability 0.13. 
* Node sizes: 100, 1000, 3000, 5000, 10,000. 

The figure below shows a simulated data set generated from SBW with the setting described above and a node size of 10,000. 
![SBM 10,000 nodes](https://github.com/xihan-qin/GEE_Sparse/blob/main/test_results/SBM_10%2C000.png)

## Comparison results
![SBM 10,000 nodes](https://github.com/xihan-qin/GEE_Sparse/blob/main/test_results/GEE_vs_GEE_sparse.png)
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



<table>
<thead>
  <tr>
    <th rowspan="2">Data Set</th>
    <th colspan="2">Lap = F, Diag = T, Cor = T</th>
    <th colspan="2">Lap = F, Diag = T, Cor = F</th>
    <th colspan="2">Lap = F, Diag = F, Cor = T</th>
    <th colspan="2">Lap = F, Diag = F, Cor = F</th>
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
    <td><strong>0.024</strong></td>
    <td>0.031</td>
    <td><strong>0.024</strong></td>
    <td>0.033</td>
    <td><strong>0.016</strong></td>    
    <td>0.034</td>
    <td><strong>0.014</strong></td>
    <td>0.031</td>
  </tr> 
  <tr>
    <td>Cora</td>
    <td><strong>0.024</strong></td>
    <td>0.025</td>
    <td><strong>0.026</strong></td>    
    <td><strong>0.026</strong></td>   
    <td><strong>0.023</strong></td>   
    <td>0.024</td>
    <td><strong>0.019</strong></td>      
    <td>0.025</td>
  </tr>
  <tr>
    <td>Proteins-all</td>
    <td>0.830</td>
    <td><strong>0.399</strong></td>      
    <td>0.623</td>
    <td><strong>0.411</strong></td>      
    <td>1.143</td>
    <td><strong>0.432</strong></td>      
    <td>0.518</td>
    <td><strong>0.462</strong></td>      
  </tr>
  <tr>
    <td>PubMed</td>
    <td>0.231</td>
    <td><strong>0.201</strong></td>        
    <td>0.201</td>
    <td>0.228</td>
    <td><strong>0.185</strong></td>    
    <td><strong>0.170</strong></td>    
    <td>0.188</td>
    <td><strong>0.177</strong></td>    
    <td>0.183</td>
  </tr>
  <tr>
    <td>CL-100K-1d8-L9</td>
    <td>1.330</td>
    <td><strong>0.909</strong></td>        
    <td>1.322</td>
    <td><strong>0.936</strong></td>    
    <td>1.114</td>
    <td><strong>0.924</strong></td>    
    <td><strong>1.058</strong></td>    
    <td>1.360</td>
  </tr>
  <tr>
    <td>CL-100K-1d8-L5</td>
    <td>203.860</td>
    <td><strong>108.977</strong></td>       
    <td>192.780</td>
    <td><strong>132.160</strong></td>       
    <td>171.838</td>
    <td><strong>125.935</strong></td>       
    <td>171.714</td>
    <td><strong>106.264</strong></td>   
  </tr>
</tbody>
</table>


