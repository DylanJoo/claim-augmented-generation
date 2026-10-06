                                                                    A      B    diff       t      p    sig  win  loss  tie   n
sys      base                target              metric                                                                       
neuclir1 qwen3-doc           qwen3-hybrid        alpha_nDCG@10 0.5642 0.6482  0.0840  2.6823 0.0152   True   16     3    0  19
                                                 alpha_nDCG@20 0.5958 0.6802  0.0843  2.9465 0.0086   True   15     4    0  19
                                                 StRecall@10   0.6963 0.7395  0.0432  2.3739 0.0289   True    9     2    8  19
                                                 StRecall@20   0.7802 0.8255  0.0453  2.1831 0.0425   True    8     3    8  19
                             qwen3-cckmeans      alpha_nDCG@10 0.5642 0.6342  0.0700  2.7964 0.0119   True   12     7    0  19
                                                 alpha_nDCG@20 0.5958 0.6611  0.0653  2.9989 0.0077   True   12     7    0  19
                                                 StRecall@10   0.6963 0.7223  0.0260  1.0069 0.3273  False    5     2   12  19
                                                 StRecall@20   0.7802 0.7902  0.0100  1.2196 0.2384  False    2     0   17  19
                             qwen3-dice          alpha_nDCG@10 0.5642 0.6923  0.1281  3.7177 0.0016   True   13     6    0  19
                                                 alpha_nDCG@20 0.5958 0.7124  0.1166  3.6970 0.0016   True   13     6    0  19
                                                 StRecall@10   0.6963 0.7844  0.0880  3.9117 0.0010   True   13     2    4  19
                                                 StRecall@20   0.7802 0.8453  0.0651  3.4836 0.0027   True   11     0    8  19
         qwen3-hybrid        qwen3-cckmeans      alpha_nDCG@10 0.6482 0.6342 -0.0141 -0.3750 0.7120  False    9    10    0  19
                                                 alpha_nDCG@20 0.6802 0.6611 -0.0191 -0.5853 0.5656  False    8    11    0  19
                                                 StRecall@10   0.7395 0.7223 -0.0172 -0.5536 0.5867  False    5     6    8  19
                                                 StRecall@20   0.8255 0.7902 -0.0353 -2.1299 0.0472   True    3     8    8  19
                             qwen3-dice          alpha_nDCG@10 0.6482 0.6923  0.0441  1.7513 0.0969  False   12     7    0  19
                                                 alpha_nDCG@20 0.6802 0.7124  0.0322  1.3201 0.2033  False   12     7    0  19
                                                 StRecall@10   0.7395 0.7844  0.0449  2.1182 0.0483   True    9     4    6  19
                                                 StRecall@20   0.8255 0.8453  0.0198  1.8219 0.0851  False    4     0   15  19
         qwen3-cckmeans      qwen3-dice          alpha_nDCG@10 0.6342 0.6923  0.0581  1.9579 0.0659  False   10     9    0  19
                                                 alpha_nDCG@20 0.6611 0.7124  0.0513  1.9903 0.0620  False   10     9    0  19
                                                 StRecall@10   0.7223 0.7844  0.0621  2.3874 0.0281   True   11     2    6  19
                                                 StRecall@20   0.7902 0.8453  0.0551  3.8255 0.0012   True   10     1    8  19
         modernbert-doc      modernbert-hybrid   alpha_nDCG@10 0.5711 0.6492  0.0781  2.6310 0.0170   True   13     6    0  19
                                                 alpha_nDCG@20 0.5996 0.6705  0.0708  2.7366 0.0136   True   14     5    0  19
                                                 StRecall@10   0.6879 0.6986  0.0107  0.3041 0.7646  False    8     6    5  19
                                                 StRecall@20   0.7890 0.7939  0.0049  0.1461 0.8855  False   10     5    4  19
                             modernbert-cckmeans alpha_nDCG@10 0.5711 0.5404 -0.0307 -0.7503 0.4628  False    7    12    0  19
                                                 alpha_nDCG@20 0.5996 0.5769 -0.0228 -0.6457 0.5266  False    8    11    0  19
                                                 StRecall@10   0.6879 0.7242  0.0363  1.0696 0.2989  False    7     6    6  19
                                                 StRecall@20   0.7890 0.8249  0.0359  1.6639 0.1135  False    6     2   11  19
                             modernbert-dice     alpha_nDCG@10 0.5711 0.6320  0.0609  1.3855 0.1828  False   12     7    0  19
                                                 alpha_nDCG@20 0.5996 0.6516  0.0520  1.3493 0.1940  False   13     6    0  19
                                                 StRecall@10   0.6879 0.7366  0.0488  1.0286 0.3173  False   10     4    5  19
                                                 StRecall@20   0.7890 0.8086  0.0195  0.4638 0.6483  False   11     5    3  19
         modernbert-hybrid   modernbert-cckmeans alpha_nDCG@10 0.6492 0.5404 -0.1088 -2.2667 0.0360   True    5    14    0  19
                                                 alpha_nDCG@20 0.6705 0.5769 -0.0936 -2.1884 0.0421   True    5    14    0  19
                                                 StRecall@10   0.6986 0.7242  0.0256  0.5429 0.5939  False    6    11    2  19
                                                 StRecall@20   0.7939 0.8249  0.0310  0.8417 0.4110  False    7     6    6  19
                             modernbert-dice     alpha_nDCG@10 0.6492 0.6320 -0.0172 -0.4165 0.6820  False    7    12    0  19
                                                 alpha_nDCG@20 0.6705 0.6516 -0.0188 -0.4937 0.6275  False    7    12    0  19
                                                 StRecall@10   0.6986 0.7366  0.0381  1.0665 0.3003  False    7     5    7  19
                                                 StRecall@20   0.7939 0.8086  0.0146  0.5565 0.5847  False    3     4   12  19
         modernbert-cckmeans modernbert-dice     alpha_nDCG@10 0.5404 0.6320  0.0916  2.5995 0.0181   True   13     6    0  19
                                                 alpha_nDCG@20 0.5769 0.6516  0.0748  2.5424 0.0204   True   12     7    0  19
                                                 StRecall@10   0.7242 0.7366  0.0124  0.3318 0.7439  False   10     5    4  19
                                                 StRecall@20   0.8249 0.8086 -0.0164 -0.4597 0.6512  False    6     6    7  19
ragtime1 qwen3-doc           qwen3-hybrid        alpha_nDCG@10 0.5598 0.5747  0.0149  1.3544 0.1848  False   19    14    1  34
                                                 alpha_nDCG@20 0.5843 0.6054  0.0211  2.0995 0.0435   True   22    12    0  34
                                                 StRecall@10   0.6958 0.7192  0.0234  1.2228 0.2301  False   13     9   12  34
                                                 StRecall@20   0.7700 0.8023  0.0323  2.5601 0.0152   True    9     2   23  34
                             qwen3-cckmeans      alpha_nDCG@10 0.5598 0.5942  0.0344  2.1292 0.0408   True   23    11    0  34
                                                 alpha_nDCG@20 0.5843 0.6169  0.0326  2.2061 0.0344   True   23    11    0  34
                                                 StRecall@10   0.6958 0.6894 -0.0064 -0.4405 0.6625  False    6     5   23  34
                                                 StRecall@20   0.7700 0.7732  0.0032  1.4312 0.1618  False    2     0   32  34
                             qwen3-dice          alpha_nDCG@10 0.5598 0.6119  0.0521  2.5908 0.0141   True   22    12    0  34
                                                 alpha_nDCG@20 0.5843 0.6347  0.0505  2.7979 0.0085   True   23    11    0  34
                                                 StRecall@10   0.6958 0.7211  0.0253  1.3890 0.1741  False   11     7   16  34
                                                 StRecall@20   0.7700 0.7962  0.0262  2.8803 0.0069   True    9     1   24  34
         qwen3-hybrid        qwen3-cckmeans      alpha_nDCG@10 0.5747 0.5942  0.0195  0.9291 0.3596  False   19    15    0  34
                                                 alpha_nDCG@20 0.6054 0.6169  0.0115  0.6261 0.5355  False   20    14    0  34
                                                 StRecall@10   0.7192 0.6894 -0.0298 -1.3244 0.1945  False    9    12   13  34
                                                 StRecall@20   0.8023 0.7732 -0.0291 -2.3269 0.0263   True    2     9   23  34
                             qwen3-dice          alpha_nDCG@10 0.5747 0.6119  0.0372  1.5745 0.1249  False   21    13    0  34
                                                 alpha_nDCG@20 0.6054 0.6347  0.0294  1.4140 0.1667  False   22    12    0  34
                                                 StRecall@10   0.7192 0.7211  0.0020  0.1311 0.8965  False    7     8   19  34
                                                 StRecall@20   0.8023 0.7962 -0.0061 -0.6785 0.5022  False    2     2   30  34
         qwen3-cckmeans      qwen3-dice          alpha_nDCG@10 0.5942 0.6119  0.0177  1.0398 0.3060  False   19    15    0  34
                                                 alpha_nDCG@20 0.6169 0.6347  0.0178  1.1505 0.2582  False   21    13    0  34
                                                 StRecall@10   0.6894 0.7211  0.0318  1.7446 0.0904  False   12     7   15  34
                                                 StRecall@20   0.7732 0.7962  0.0230  2.6678 0.0117   True    9     2   23  34
         modernbert-doc      modernbert-hybrid   alpha_nDCG@10 0.5230 0.5831  0.0601  2.9077 0.0065   True   24    10    0  34
                                                 alpha_nDCG@20 0.5491 0.6160  0.0670  3.2965 0.0023   True   24    10    0  34
                                                 StRecall@10   0.6592 0.6933  0.0341  1.8088 0.0796  False   12     5   17  34
                                                 StRecall@20   0.7364 0.7958  0.0594  2.4287 0.0208   True   12     5   17  34
                             modernbert-cckmeans alpha_nDCG@10 0.5230 0.5104 -0.0125 -0.5466 0.5883  False   14    20    0  34
                                                 alpha_nDCG@20 0.5491 0.5431 -0.0060 -0.3003 0.7658  False   14    20    0  34
                                                 StRecall@10   0.6592 0.6448 -0.0144 -0.8196 0.4183  False    8     6   20  34
                                                 StRecall@20   0.7364 0.7518  0.0153  0.9004 0.3745  False    4     3   27  34
                             modernbert-dice     alpha_nDCG@10 0.5230 0.5634  0.0404  1.7538 0.0888  False   22    12    0  34
                                                 alpha_nDCG@20 0.5491 0.5852  0.0362  1.7707 0.0858  False   20    14    0  34
                                                 StRecall@10   0.6592 0.7143  0.0551  1.9610 0.0584  False   15     7   12  34
                                                 StRecall@20   0.7364 0.7953  0.0589  2.2585 0.0306   True    9     7   18  34
         modernbert-hybrid   modernbert-cckmeans alpha_nDCG@10 0.5831 0.5104 -0.0726 -2.1931 0.0355   True   12    22    0  34
                                                 alpha_nDCG@20 0.6160 0.5431 -0.0730 -2.5750 0.0147   True   13    21    0  34
                                                 StRecall@10   0.6933 0.6448 -0.0485 -1.9839 0.0556  False    6    13   15  34
                                                 StRecall@20   0.7958 0.7518 -0.0441 -2.2214 0.0333   True    6    11   17  34
                             modernbert-dice     alpha_nDCG@10 0.5831 0.5634 -0.0197 -0.7869 0.4370  False   15    19    0  34
                                                 alpha_nDCG@20 0.6160 0.5852 -0.0308 -1.3929 0.1730  False   14    20    0  34
                                                 StRecall@10   0.6933 0.7143  0.0210  0.9725 0.3379  False   12     9   13  34
                                                 StRecall@20   0.7958 0.7953 -0.0005 -0.0630 0.9501  False    4     6   24  34
         modernbert-cckmeans modernbert-dice     alpha_nDCG@10 0.5104 0.5634  0.0529  2.7628 0.0093   True   21    13    0  34
                                                 alpha_nDCG@20 0.5431 0.5852  0.0421  2.6133 0.0134   True   23    11    0  34
                                                 StRecall@10   0.6448 0.7143  0.0695  2.5264 0.0165   True   15     8   11  34
                                                 StRecall@20   0.7518 0.7953  0.0436  2.0161 0.0520  False   10     8   16  34
