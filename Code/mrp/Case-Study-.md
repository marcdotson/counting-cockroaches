MRP Case Study
================
McKenna Weech
1/17/2020

Firms employ a variety of passive listening platforms (e.g., customer
comment cards, help lines, social media) that elicit a variety of
complaints. Since these complaints are self-selected, it isn’t obvious
if complaints are widespread or genuinely indicative of systemic service
failures. In other words, how can can a firm know when complaints can be
ignored and when they need to addressed? This issue is related to a
larger class of problems where the goal is to make inference about the
size of a population from a non-random sample (e.g., the German tank
problem). In this paper, we develop a model that uses a variety of
observed customer complaints on Twitter to make inference about the
severity of service failures.

## Describe the model conceptually

It has been said that if you observe a cockroach on your floor, there
are likely thousands inside the walls of your home. Following this
analogy, our hope is to count the cockroaches we can see and calculate
how many are hiding in the walls.

For the first portion of this case, we will be making a model around
simulated data. Each observation will represent a single tweet, and we
will be scoring them for satisfaction. We will use a multiple hierarchal
linear model to predict satisfaction from the other known variables.
Once we have the model predicting well on our simulated data, we will
fit it to our real data, and from there use our real data satisfaction
scores to measure service or product failure severity.

Along with making a model, we will also be using poststratification to
correct the weights in our simulated sample and real data to more
correctly represent the population. This will be explained in greater
detail as we postratify in this first iteration of the model, so we can
explain and see the benefit of a full multiple regression with
postratification model (MRP).

To begin, we will start by generating our simulated data. It will run
through an initial simple linear model and postratification.

# Iteration 1: Simulation Data and Simple Model

This initial block of code, we will just be loading in the needed
packages and settings to make our sample and model code run.

``` r
library(tidyverse) #Load packages
library(rstan)
library(rstanarm)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default())
library(dplyr)
library(tidyr)
library(tidybayes)

options(mc.cores = parallel::detectCores()) # Set Stan to use all availible cores
rstan_options(auto_write = TRUE) # Don't recompile Stan code that hasn't changed
```

For this first iteration of modeling, we will be generating our own
data. We will generate a population, and sample from it in order to
build an MRP model.

We will be modeling our populations `satisfaction`. This will eventually
translate into measuring service severity when we apply our model to our
real data. We will be looking at common characteristics like `gender`,
`ethnicity`, `income`, `age`, and `state`.

The following code will make a function `simulate_mrp_data` that will
help us make our simulated data, and sample from it in such a way that
we can use and see the value in postratification.

``` r
simulate_mrp_data <- function(n) {
  J <- c(2, 3, 7, 3, 50) # male or not, eth, age, income level, state
  poststrat <- as.data.frame(array(NA, c(prod(J), length(J)+1))) # Columns of post-strat matrix, plus one for size
  colnames(poststrat) <- c("male", "eth", "age","income", "state",'N')
  count <- 0
  for (i1 in 1:J[1]){
    for (i2 in 1:J[2]){
      for (i3 in 1:J[3]){
        for (i4 in 1:J[4]){
          for (i5 in 1:J[5]){
              count <- count + 1
              # Fill them in so we know what category we are referring to
              poststrat[count, 1:5] <- c(i1-1, i2, i3, i4, i5)
          }
        }
      }
    }
  }
  # Proportion in each sample in the population
  p_male <- c(0.52, 0.48)
  p_eth <- c(0.5, 0.2, 0.3)
  p_age <- c(0.2,.1,0.2,0.2, 0.10, 0.1, 0.1)
  p_income<-c(.50,.35,.15)
  p_state_tmp<-runif(50,10,20)
  p_state<-p_state_tmp/sum(p_state_tmp)
  poststrat$N<-0
  for (j in 1:prod(J)){
    poststrat$N[j] <- round(250e6 * p_male[poststrat[j,1]+1] * p_eth[poststrat[j,2]] *
      p_age[poststrat[j,3]]*p_income[poststrat[j,4]]*p_state[poststrat[j,5]]) #Adjust the N to be the number observed in each category in each group
  }

  # Now let's adjust for the probability of response
  p_response_baseline <- 0.01
  p_response_male <- c(2, 0.8) / 2.8
  p_response_eth <- c(1, 1.2, 2.5) / 4.7
  p_response_age <- c(1, 0.4, 1, 1.5,  3, 5, 7) / 18.9
  p_response_inc <- c(1, 0.9, 0.8) / 2.7
  p_response_state <- rbeta(50, 1, 1)
  p_response_state <- p_response_state / sum(p_response_state)
  p_response <- rep(NA, prod(J))
  for (j in 1:prod(J)) {
    p_response[j] <-
      p_response_baseline * p_response_male[poststrat[j, 1] + 1] *
      p_response_eth[poststrat[j, 2]] * p_response_age[poststrat[j, 3]] *
      p_response_inc[poststrat[j, 4]] * p_response_state[poststrat[j, 5]]
  }
  people <- sample(prod(J), n, replace = TRUE, prob = poststrat$N * p_response)

  ## For respondent i, people[i] is that person's poststrat cell,
  ## some number between 1 and 32
  n_cell <- rep(NA, prod(J))
  for (j in 1:prod(J)) {
    n_cell[j] <- sum(people == j)
  }

  coef_male <- c(0,-0.3)
  coef_eth <- c(0, 0.6, 0.9)
  coef_age <- c(0,-0.2,-0.3, 0.4, 0.5, 0.7, 0.8, 0.9)
  coef_income <- c(0,-0.2, 0.6)
  coef_state <- c(0, round(rnorm(49, 0, 1), 1))
  coef_age_male <- t(cbind(c(0, .1, .23, .3, .43, .5, .6),
                           c(0, -.1, -.23, -.5, -.43, -.5, -.6)))
  true_popn <- data.frame(poststrat[, 1:5], service_failure = rep(NA, prod(J)))
  for (j in 1:prod(J)) {
    true_popn$satisfaction[j] <- plogis(
      coef_male[poststrat[j, 1] + 1] +
        coef_eth[poststrat[j, 2]] + coef_age[poststrat[j, 3]] +
        coef_income[poststrat[j, 4]] + coef_state[poststrat[j, 5]] +
        coef_age_male[poststrat[j, 1] + 1, poststrat[j, 3]]
      )
  }

  #male or not, eth, age, income level, state, city
  y <- rbinom(n, 1, true_popn$satisfaction[people])
  male <- poststrat[people, 1]
  eth <- poststrat[people, 2]
  age <- poststrat[people, 3]
  income <- poststrat[people, 4]
  state <- poststrat[people, 5]

  sample <- data.frame(service_failure = y,
                       male, age, eth, income, state,
                       id = 1:length(people))

  #Make all numeric:
  for (i in 1:ncol(poststrat)) {
    poststrat[, i] <- as.numeric(poststrat[, i])
  }
  for (i in 1:ncol(true_popn)) {
    true_popn[, i] <- as.numeric(true_popn[, i])
  }
  for (i in 1:ncol(sample)) {
    sample[, i] <- as.numeric(sample[, i])
  }
  list(
    sample = sample,
    poststrat = poststrat,
    true_popn = true_popn
  )
}
```

``` r
mrp_sim <- simulate_mrp_data(n=1200)
sample <- mrp_sim[["sample"]]

print(sample)
```

    ##      service_failure male age eth income state   id
    ## 1                  1    0   7   3      2    33    1
    ## 2                  0    1   4   1      3    44    2
    ## 3                  1    0   1   2      1     6    3
    ## 4                  1    0   5   1      3    47    4
    ## 5                  1    0   6   3      1    45    5
    ## 6                  1    0   6   3      3    21    6
    ## 7                  1    0   7   3      1    36    7
    ## 8                  1    0   7   2      3    22    8
    ## 9                  1    0   6   2      3    35    9
    ## 10                 1    1   5   2      1    45   10
    ## 11                 0    0   5   1      2    28   11
    ## 12                 0    0   6   3      1    41   12
    ## 13                 1    0   7   1      1    28   13
    ## 14                 1    0   6   3      2    45   14
    ## 15                 0    1   1   3      1     6   15
    ## 16                 1    0   4   2      2    35   16
    ## 17                 1    0   6   3      1    20   17
    ## 18                 1    0   7   1      3    49   18
    ## 19                 1    1   6   1      2    23   19
    ## 20                 0    0   5   3      1     7   20
    ## 21                 0    0   6   2      1     7   21
    ## 22                 0    1   3   3      1    22   22
    ## 23                 1    1   5   3      1    38   23
    ## 24                 1    0   7   3      1    36   24
    ## 25                 1    0   7   1      3    47   25
    ## 26                 1    1   7   3      3    12   26
    ## 27                 0    0   7   3      1    20   27
    ## 28                 0    0   5   3      1     2   28
    ## 29                 0    0   6   3      1    11   29
    ## 30                 1    0   4   3      2    24   30
    ## 31                 1    1   7   3      1    42   31
    ## 32                 1    1   1   2      3    24   32
    ## 33                 1    0   4   3      1    13   33
    ## 34                 1    0   7   3      1    41   34
    ## 35                 1    0   6   3      1    22   35
    ## 36                 1    1   7   1      3     1   36
    ## 37                 1    0   3   3      1    31   37
    ## 38                 1    0   7   3      3    12   38
    ## 39                 1    0   7   1      3    11   39
    ## 40                 0    0   3   3      3    21   40
    ## 41                 1    0   1   3      3    26   41
    ## 42                 0    1   3   3      3    44   42
    ## 43                 1    0   4   1      1    22   43
    ## 44                 1    0   7   1      2    19   44
    ## 45                 0    0   1   1      1     7   45
    ## 46                 1    0   4   1      1    28   46
    ## 47                 0    0   4   3      1    17   47
    ## 48                 1    0   6   3      2    11   48
    ## 49                 1    0   7   1      1    38   49
    ## 50                 1    0   7   3      1    47   50
    ## 51                 0    1   6   1      1     6   51
    ## 52                 1    0   7   1      1    47   52
    ## 53                 1    1   5   3      2    30   53
    ## 54                 1    1   4   2      1     1   54
    ## 55                 0    0   5   3      2    22   55
    ## 56                 1    0   4   3      1    10   56
    ## 57                 1    0   4   1      1    39   57
    ## 58                 1    1   5   3      1    10   58
    ## 59                 1    0   7   3      1    14   59
    ## 60                 0    1   6   1      1    23   60
    ## 61                 1    0   7   2      1    36   61
    ## 62                 1    1   3   3      1    39   62
    ## 63                 1    0   3   3      2    42   63
    ## 64                 1    0   4   2      3    28   64
    ## 65                 1    1   3   1      2    47   65
    ## 66                 1    0   3   2      2    45   66
    ## 67                 1    0   6   3      2    23   67
    ## 68                 1    0   6   3      1    30   68
    ## 69                 1    0   3   1      2    28   69
    ## 70                 0    1   7   1      3    26   70
    ## 71                 1    1   4   3      2    34   71
    ## 72                 0    0   4   3      1    27   72
    ## 73                 1    0   7   3      2    23   73
    ## 74                 1    0   6   2      1    20   74
    ## 75                 1    0   7   2      2    47   75
    ## 76                 0    0   1   3      1    34   76
    ## 77                 1    1   6   3      1    44   77
    ## 78                 1    1   7   3      2    47   78
    ## 79                 0    0   5   1      2     4   79
    ## 80                 1    0   7   3      2     1   80
    ## 81                 1    0   1   2      2     1   81
    ## 82                 0    0   5   1      1    48   82
    ## 83                 1    0   2   3      1    41   83
    ## 84                 0    1   7   3      1     4   84
    ## 85                 1    1   7   3      3     7   85
    ## 86                 1    0   7   3      2    48   86
    ## 87                 1    0   7   3      2    37   87
    ## 88                 0    0   7   2      3    30   88
    ## 89                 1    0   4   1      1    45   89
    ## 90                 1    1   1   2      1    19   90
    ## 91                 1    0   5   3      1    28   91
    ## 92                 1    1   7   2      2     5   92
    ## 93                 1    1   5   3      2    35   93
    ## 94                 1    0   6   3      1    42   94
    ## 95                 1    0   4   3      1    30   95
    ## 96                 1    0   6   2      1    13   96
    ## 97                 1    1   7   3      1    44   97
    ## 98                 1    0   7   1      3    29   98
    ## 99                 0    1   3   1      2    44   99
    ## 100                0    0   1   3      2    28  100
    ## 101                1    1   7   1      1     1  101
    ## 102                1    1   3   3      3    43  102
    ## 103                1    0   7   3      1    35  103
    ## 104                1    0   6   1      1    29  104
    ## 105                1    1   7   3      2     1  105
    ## 106                1    0   4   2      1    19  106
    ## 107                1    0   4   3      1    45  107
    ## 108                0    1   7   1      2    30  108
    ## 109                1    0   7   3      1    22  109
    ## 110                1    0   4   3      2    13  110
    ## 111                1    0   4   3      1    42  111
    ## 112                1    0   3   1      2     1  112
    ## 113                1    1   6   2      1    47  113
    ## 114                1    0   7   3      1    12  114
    ## 115                1    0   7   2      2    39  115
    ## 116                1    1   7   1      1    47  116
    ## 117                1    0   7   1      1    20  117
    ## 118                1    0   7   3      2    35  118
    ## 119                1    0   1   1      2    10  119
    ## 120                0    0   3   3      1    12  120
    ## 121                1    0   7   1      1    32  121
    ## 122                0    0   7   1      1    12  122
    ## 123                0    0   5   1      2    12  123
    ## 124                0    0   5   3      1    43  124
    ## 125                0    1   6   1      3    39  125
    ## 126                1    0   3   1      2    44  126
    ## 127                1    0   5   3      2    42  127
    ## 128                0    0   1   3      2    12  128
    ## 129                1    1   7   1      2    42  129
    ## 130                0    1   7   3      1     7  130
    ## 131                1    0   7   3      1    19  131
    ## 132                1    0   3   1      3    31  132
    ## 133                0    0   4   1      2    30  133
    ## 134                1    0   2   3      2    35  134
    ## 135                1    0   1   1      1    34  135
    ## 136                0    0   7   3      1    12  136
    ## 137                0    1   7   1      2     1  137
    ## 138                1    0   7   2      2    49  138
    ## 139                0    0   2   1      1    41  139
    ## 140                1    0   7   3      1    41  140
    ## 141                1    0   7   3      2    36  141
    ## 142                1    1   1   2      1     4  142
    ## 143                1    1   7   2      3    36  143
    ## 144                1    0   6   2      1    49  144
    ## 145                1    1   6   2      1    39  145
    ## 146                1    0   7   3      2    21  146
    ## 147                1    0   7   1      1    41  147
    ## 148                1    0   7   3      1    22  148
    ## 149                1    1   5   3      1    44  149
    ## 150                1    0   5   1      1    26  150
    ## 151                1    1   7   3      1    45  151
    ## 152                1    0   6   1      1    31  152
    ## 153                0    0   4   1      3    44  153
    ## 154                1    0   7   3      3    35  154
    ## 155                0    0   7   3      2    49  155
    ## 156                1    0   4   3      1    12  156
    ## 157                1    0   6   1      1    31  157
    ## 158                0    1   5   3      1    41  158
    ## 159                1    0   1   3      3    14  159
    ## 160                1    0   1   3      3    16  160
    ## 161                1    0   3   3      1    44  161
    ## 162                1    1   1   1      1    39  162
    ## 163                1    0   6   1      1     7  163
    ## 164                1    0   6   3      1    11  164
    ## 165                1    0   6   3      2    38  165
    ## 166                1    0   4   1      2    38  166
    ## 167                1    0   3   1      1    41  167
    ## 168                1    0   1   3      2    23  168
    ## 169                1    0   7   3      1    14  169
    ## 170                1    0   3   3      1    24  170
    ## 171                1    0   5   3      1    36  171
    ## 172                1    0   7   1      1    14  172
    ## 173                0    1   7   1      1    49  173
    ## 174                1    1   6   1      1    28  174
    ## 175                1    0   4   1      3    35  175
    ## 176                1    0   6   3      2    31  176
    ## 177                1    0   7   3      1    28  177
    ## 178                0    0   5   1      1    22  178
    ## 179                1    0   6   1      2    43  179
    ## 180                0    0   1   3      1    12  180
    ## 181                0    0   4   1      2     4  181
    ## 182                1    0   4   3      2     3  182
    ## 183                0    0   4   3      1    12  183
    ## 184                1    1   7   1      2    36  184
    ## 185                1    0   6   3      1     6  185
    ## 186                0    0   6   1      1    26  186
    ## 187                1    1   4   3      1    33  187
    ## 188                1    0   7   1      1    38  188
    ## 189                1    0   6   3      1    30  189
    ## 190                0    0   6   1      1    41  190
    ## 191                1    0   3   3      2    27  191
    ## 192                0    1   5   1      2    10  192
    ## 193                0    1   1   3      1    30  193
    ## 194                0    0   3   1      2    45  194
    ## 195                1    0   3   1      2    29  195
    ## 196                0    1   6   3      1    12  196
    ## 197                0    0   4   3      2    22  197
    ## 198                1    1   6   3      3    35  198
    ## 199                1    0   7   1      1    19  199
    ## 200                0    1   6   1      3     1  200
    ## 201                0    1   5   3      2    22  201
    ## 202                1    0   7   1      1    13  202
    ## 203                1    0   4   2      2     4  203
    ## 204                1    0   4   1      1    14  204
    ## 205                0    0   6   1      2    22  205
    ## 206                1    0   6   2      2    30  206
    ## 207                0    1   1   2      1    47  207
    ## 208                1    1   4   3      1    29  208
    ## 209                1    0   6   3      3    19  209
    ## 210                1    0   4   3      2     6  210
    ## 211                1    0   6   3      3    45  211
    ## 212                0    0   3   2      2    26  212
    ## 213                1    0   4   3      2    47  213
    ## 214                1    1   7   2      2    11  214
    ## 215                1    0   7   3      2    46  215
    ## 216                0    0   3   1      2     1  216
    ## 217                1    0   7   3      2    27  217
    ## 218                1    1   5   3      2    45  218
    ## 219                0    1   4   3      1    41  219
    ## 220                0    1   5   3      1    41  220
    ## 221                0    1   7   2      1    24  221
    ## 222                1    0   4   3      1     1  222
    ## 223                1    0   6   3      1    47  223
    ## 224                1    0   6   2      1    14  224
    ## 225                1    1   1   3      3    33  225
    ## 226                1    0   6   3      1    17  226
    ## 227                1    0   3   3      1    27  227
    ## 228                1    0   7   1      1    47  228
    ## 229                1    0   1   1      1    36  229
    ## 230                0    1   1   3      3    22  230
    ## 231                1    0   6   1      2    27  231
    ## 232                1    0   5   3      2    47  232
    ## 233                1    0   7   3      1    41  233
    ## 234                1    0   7   3      2    36  234
    ## 235                1    0   4   3      2     3  235
    ## 236                1    0   7   1      3    22  236
    ## 237                1    0   6   3      1    38  237
    ## 238                1    0   6   3      1    42  238
    ## 239                1    1   7   3      3    22  239
    ## 240                0    1   5   2      2    41  240
    ## 241                0    0   3   3      2     9  241
    ## 242                1    0   4   3      3    33  242
    ## 243                1    1   7   3      1    41  243
    ## 244                1    0   6   3      1    44  244
    ## 245                1    0   4   3      1    23  245
    ## 246                0    1   6   1      2    27  246
    ## 247                1    0   7   2      1    47  247
    ## 248                1    0   6   3      2    10  248
    ## 249                0    1   4   3      2    30  249
    ## 250                0    1   4   1      3    12  250
    ## 251                0    1   7   1      2    30  251
    ## 252                1    0   7   1      3    31  252
    ## 253                1    0   4   1      3    14  253
    ## 254                1    0   5   2      2    31  254
    ## 255                0    0   6   1      1    48  255
    ## 256                0    0   2   1      1     1  256
    ## 257                1    0   1   3      2     1  257
    ## 258                1    0   7   3      2     1  258
    ## 259                1    1   6   3      1    38  259
    ## 260                1    0   5   1      1    47  260
    ## 261                1    0   4   1      1    45  261
    ## 262                1    0   6   2      1     6  262
    ## 263                1    0   1   3      2     1  263
    ## 264                1    0   6   3      2    31  264
    ## 265                1    0   1   3      2    47  265
    ## 266                1    1   1   2      1    29  266
    ## 267                0    0   7   1      1    26  267
    ## 268                1    0   7   1      1    36  268
    ## 269                0    1   5   1      3     4  269
    ## 270                1    0   4   3      1     1  270
    ## 271                0    0   6   1      2     6  271
    ## 272                0    0   6   1      1    12  272
    ## 273                1    1   7   3      1    19  273
    ## 274                1    0   5   1      2    36  274
    ## 275                1    0   7   2      1    38  275
    ## 276                0    0   3   1      1    12  276
    ## 277                1    0   6   3      3    23  277
    ## 278                0    0   3   3      3     7  278
    ## 279                0    0   3   3      1    47  279
    ## 280                1    0   4   2      1    41  280
    ## 281                1    0   5   3      3    10  281
    ## 282                0    1   6   3      2     9  282
    ## 283                1    0   1   3      2    42  283
    ## 284                1    0   4   3      2    45  284
    ## 285                1    0   5   3      3    10  285
    ## 286                1    0   7   3      2    34  286
    ## 287                1    1   5   3      1    38  287
    ## 288                0    1   7   1      1    11  288
    ## 289                1    1   7   3      1    11  289
    ## 290                0    0   6   2      2     5  290
    ## 291                1    1   7   1      2    47  291
    ## 292                1    1   7   3      2     1  292
    ## 293                1    1   1   3      1    22  293
    ## 294                1    0   7   2      1    11  294
    ## 295                0    1   7   2      2     2  295
    ## 296                0    0   6   1      1    10  296
    ## 297                1    1   6   2      2    41  297
    ## 298                1    0   7   1      3    14  298
    ## 299                1    1   7   3      2    29  299
    ## 300                1    0   7   3      3     4  300
    ## 301                1    0   6   1      1    23  301
    ## 302                0    0   2   2      1    20  302
    ## 303                1    0   7   2      2     3  303
    ## 304                1    0   1   3      2    39  304
    ## 305                1    0   7   3      1    38  305
    ## 306                1    1   5   1      2    35  306
    ## 307                1    0   7   3      2    14  307
    ## 308                1    0   6   1      1     6  308
    ## 309                0    0   3   3      2    26  309
    ## 310                1    0   7   2      1    35  310
    ## 311                1    0   5   1      1     6  311
    ## 312                1    0   7   3      2    30  312
    ## 313                1    0   6   3      1    11  313
    ## 314                0    1   3   3      3    11  314
    ## 315                1    0   5   3      1    43  315
    ## 316                1    0   7   1      1    35  316
    ## 317                1    0   7   1      1    29  317
    ## 318                1    0   7   3      1    41  318
    ## 319                1    0   7   1      3    44  319
    ## 320                0    1   7   3      1    39  320
    ## 321                1    0   5   3      1    22  321
    ## 322                0    0   5   3      3    22  322
    ## 323                1    1   3   1      2     4  323
    ## 324                0    0   3   1      2    29  324
    ## 325                1    0   7   1      1    36  325
    ## 326                1    0   7   1      2    12  326
    ## 327                1    0   4   3      2    45  327
    ## 328                1    0   5   2      1    44  328
    ## 329                1    1   6   1      1     1  329
    ## 330                1    0   4   2      1    43  330
    ## 331                0    0   4   2      2    12  331
    ## 332                1    0   5   1      1    42  332
    ## 333                1    0   4   1      1    36  333
    ## 334                1    0   7   3      1     6  334
    ## 335                1    0   5   1      1    24  335
    ## 336                1    1   5   3      1     5  336
    ## 337                1    0   7   3      2    26  337
    ## 338                0    0   7   3      1    12  338
    ## 339                1    0   6   1      1    21  339
    ## 340                1    0   7   1      1    38  340
    ## 341                1    1   6   3      1     9  341
    ## 342                0    1   7   3      1    22  342
    ## 343                1    0   1   1      1     5  343
    ## 344                1    0   6   2      1    13  344
    ## 345                1    0   7   1      2    45  345
    ## 346                0    0   5   1      2    12  346
    ## 347                1    0   6   3      1    22  347
    ## 348                0    1   5   2      1    33  348
    ## 349                1    0   7   2      2     1  349
    ## 350                1    0   7   3      1    33  350
    ## 351                0    0   7   2      2     6  351
    ## 352                1    0   4   1      1    47  352
    ## 353                1    0   6   3      1    47  353
    ## 354                0    1   7   2      1    22  354
    ## 355                0    0   1   1      2    28  355
    ## 356                0    0   3   2      2    36  356
    ## 357                1    1   4   3      3    19  357
    ## 358                0    0   1   1      2    12  358
    ## 359                0    0   7   2      1    22  359
    ## 360                0    1   7   3      2    10  360
    ## 361                1    0   5   3      2    44  361
    ## 362                1    0   4   3      1    18  362
    ## 363                1    0   6   3      1    11  363
    ## 364                1    0   6   1      2    28  364
    ## 365                0    1   1   2      1    14  365
    ## 366                0    1   5   2      1    37  366
    ## 367                0    1   4   2      1     2  367
    ## 368                0    0   1   3      2    43  368
    ## 369                0    0   3   3      2    22  369
    ## 370                1    0   7   3      3    28  370
    ## 371                0    0   7   1      3    12  371
    ## 372                0    1   7   3      3    23  372
    ## 373                0    0   5   1      1    12  373
    ## 374                1    1   6   2      1    14  374
    ## 375                1    0   5   2      2    22  375
    ## 376                1    0   7   3      1    45  376
    ## 377                1    1   7   3      2    30  377
    ## 378                0    1   6   3      1    20  378
    ## 379                0    1   6   3      1    28  379
    ## 380                0    0   1   3      3     1  380
    ## 381                1    0   4   3      3    34  381
    ## 382                1    0   3   3      2    38  382
    ## 383                1    0   4   3      1    38  383
    ## 384                1    0   3   3      3    15  384
    ## 385                1    1   1   1      2    45  385
    ## 386                0    1   5   2      2     7  386
    ## 387                0    1   7   1      1     1  387
    ## 388                1    0   7   2      1     4  388
    ## 389                1    1   6   3      2    14  389
    ## 390                1    1   5   3      1    41  390
    ## 391                0    1   6   3      1    41  391
    ## 392                0    1   5   3      1     5  392
    ## 393                1    0   5   1      2    31  393
    ## 394                1    0   4   3      2    18  394
    ## 395                1    0   1   3      2     1  395
    ## 396                1    0   4   3      1    44  396
    ## 397                1    0   7   3      2    22  397
    ## 398                1    0   7   3      1    18  398
    ## 399                1    1   7   1      2    27  399
    ## 400                1    1   5   2      2    33  400
    ## 401                1    0   5   3      2    36  401
    ## 402                1    0   6   2      2    41  402
    ## 403                0    0   3   1      2    12  403
    ## 404                1    0   5   1      1    38  404
    ## 405                1    0   4   3      3    12  405
    ## 406                1    1   1   3      1    41  406
    ## 407                1    0   6   3      1     6  407
    ## 408                1    0   3   2      1    21  408
    ## 409                0    0   1   3      1     6  409
    ## 410                1    1   6   3      1    47  410
    ## 411                1    0   6   1      3    45  411
    ## 412                1    0   5   3      2    21  412
    ## 413                1    1   4   3      3     1  413
    ## 414                1    0   4   3      2    35  414
    ## 415                1    0   7   3      2    38  415
    ## 416                1    0   3   2      3     1  416
    ## 417                1    0   6   3      1    27  417
    ## 418                1    1   7   3      1    22  418
    ## 419                1    0   7   1      2    17  419
    ## 420                1    0   4   1      2    47  420
    ## 421                0    0   7   2      2     7  421
    ## 422                1    1   6   1      2    33  422
    ## 423                0    1   4   1      2    28  423
    ## 424                1    0   6   3      3    33  424
    ## 425                0    1   7   1      1    30  425
    ## 426                1    0   5   3      2     4  426
    ## 427                1    0   6   1      2    36  427
    ## 428                1    0   3   1      2    39  428
    ## 429                1    0   7   1      1    39  429
    ## 430                1    0   3   3      3    37  430
    ## 431                1    0   6   1      1    45  431
    ## 432                1    0   6   3      3     4  432
    ## 433                1    0   6   3      1    18  433
    ## 434                1    0   7   3      1    32  434
    ## 435                1    0   7   3      2    22  435
    ## 436                1    0   7   3      1    33  436
    ## 437                0    0   4   2      2     4  437
    ## 438                1    0   6   3      2    47  438
    ## 439                0    1   6   1      1    39  439
    ## 440                1    0   7   2      1    36  440
    ## 441                1    0   6   2      3    41  441
    ## 442                1    0   4   1      1    31  442
    ## 443                1    0   3   1      1    41  443
    ## 444                0    1   6   1      1     3  444
    ## 445                0    1   7   2      1    22  445
    ## 446                1    0   3   3      1    28  446
    ## 447                1    0   1   3      2    49  447
    ## 448                1    0   6   1      1    45  448
    ## 449                0    0   4   1      1     1  449
    ## 450                1    0   7   1      1    19  450
    ## 451                0    0   7   3      1    28  451
    ## 452                1    1   6   2      2    42  452
    ## 453                1    0   7   1      2    41  453
    ## 454                1    0   7   3      3    28  454
    ## 455                1    0   6   3      1    50  455
    ## 456                1    0   4   1      2    35  456
    ## 457                1    0   7   3      1    50  457
    ## 458                1    0   7   1      1     4  458
    ## 459                0    0   5   3      1     7  459
    ## 460                0    0   3   3      2    10  460
    ## 461                1    0   5   3      3    30  461
    ## 462                1    0   6   3      1    21  462
    ## 463                0    1   7   1      1     7  463
    ## 464                1    1   1   3      1    12  464
    ## 465                1    0   7   1      3    36  465
    ## 466                1    0   7   1      1    18  466
    ## 467                1    0   6   1      3    19  467
    ## 468                1    0   5   1      2     3  468
    ## 469                0    0   3   2      2    10  469
    ## 470                1    0   6   3      2    41  470
    ## 471                1    1   3   1      1     5  471
    ## 472                0    1   6   3      1     4  472
    ## 473                1    0   4   2      1    26  473
    ## 474                0    1   7   1      1    17  474
    ## 475                1    0   5   3      2     6  475
    ## 476                0    1   7   1      2     2  476
    ## 477                1    0   6   3      2    34  477
    ## 478                1    0   7   3      1     3  478
    ## 479                1    0   4   1      2     5  479
    ## 480                0    1   3   1      1    21  480
    ## 481                1    0   6   1      1    33  481
    ## 482                0    1   4   3      1     2  482
    ## 483                1    1   5   2      2    22  483
    ## 484                1    1   6   1      1    38  484
    ## 485                1    0   7   1      1    45  485
    ## 486                1    0   6   3      2    23  486
    ## 487                0    0   3   1      2     1  487
    ## 488                0    0   7   1      1    22  488
    ## 489                1    0   7   1      2    45  489
    ## 490                1    0   7   3      1    18  490
    ## 491                1    0   7   3      2    38  491
    ## 492                0    0   1   1      1    41  492
    ## 493                1    0   6   3      3    11  493
    ## 494                1    0   7   1      1     2  494
    ## 495                1    1   5   1      1    38  495
    ## 496                0    1   6   3      1    18  496
    ## 497                1    0   7   3      1     3  497
    ## 498                0    0   4   1      2     6  498
    ## 499                1    1   5   3      2    14  499
    ## 500                0    0   6   2      1    28  500
    ## 501                1    0   5   1      1    41  501
    ## 502                0    0   7   1      1    24  502
    ## 503                1    0   4   2      1    23  503
    ## 504                0    0   3   2      1    22  504
    ## 505                1    1   6   3      2     1  505
    ## 506                1    0   5   3      2    31  506
    ## 507                0    0   7   1      1    39  507
    ## 508                1    1   6   1      3    27  508
    ## 509                0    0   6   3      1    20  509
    ## 510                0    0   6   1      2    10  510
    ## 511                0    0   3   2      2     1  511
    ## 512                1    0   7   3      1    41  512
    ## 513                1    0   6   1      2    19  513
    ## 514                0    1   7   3      1    12  514
    ## 515                0    0   1   3      1    31  515
    ## 516                1    0   7   3      1    23  516
    ## 517                1    0   6   3      2    41  517
    ## 518                1    0   3   3      2     1  518
    ## 519                1    0   6   3      1    30  519
    ## 520                1    0   4   1      1    45  520
    ## 521                1    0   4   1      2    19  521
    ## 522                0    0   1   3      1    30  522
    ## 523                0    0   7   1      2    12  523
    ## 524                1    0   7   2      1    36  524
    ## 525                1    0   6   1      3    35  525
    ## 526                1    1   7   3      1    11  526
    ## 527                1    0   3   2      2    32  527
    ## 528                0    0   1   2      2    42  528
    ## 529                0    0   2   1      1    41  529
    ## 530                1    1   6   1      2    38  530
    ## 531                1    0   4   3      1     1  531
    ## 532                1    0   1   3      3    42  532
    ## 533                1    0   6   3      1    42  533
    ## 534                0    0   1   3      2    12  534
    ## 535                1    1   2   3      1     5  535
    ## 536                0    0   3   3      2    28  536
    ## 537                1    0   7   3      2    44  537
    ## 538                1    0   7   3      1    45  538
    ## 539                1    0   4   3      1    45  539
    ## 540                1    0   3   3      1    29  540
    ## 541                1    0   5   1      2    30  541
    ## 542                0    1   3   2      2    21  542
    ## 543                1    0   4   2      2     1  543
    ## 544                0    1   6   1      1    13  544
    ## 545                1    0   7   1      2    42  545
    ## 546                1    1   4   3      1    42  546
    ## 547                0    0   2   1      1    22  547
    ## 548                0    0   4   3      2    12  548
    ## 549                0    1   7   3      1    48  549
    ## 550                1    1   5   3      1    41  550
    ## 551                1    0   7   3      1    12  551
    ## 552                1    0   6   1      3    11  552
    ## 553                1    0   3   3      2    20  553
    ## 554                0    0   6   3      1    12  554
    ## 555                1    1   7   1      2    23  555
    ## 556                1    0   4   1      2     9  556
    ## 557                1    0   4   1      1    14  557
    ## 558                0    0   3   1      3    33  558
    ## 559                1    0   1   3      2    45  559
    ## 560                0    1   6   3      1    43  560
    ## 561                1    0   6   1      1    12  561
    ## 562                1    0   4   2      1    39  562
    ## 563                1    1   3   2      1    38  563
    ## 564                1    0   5   1      1    45  564
    ## 565                1    0   7   3      2    26  565
    ## 566                1    1   4   2      2    38  566
    ## 567                1    1   6   3      2    10  567
    ## 568                1    0   7   3      1     6  568
    ## 569                1    0   6   3      2     1  569
    ## 570                1    1   7   2      3    47  570
    ## 571                0    0   6   1      1    41  571
    ## 572                1    0   5   3      2     4  572
    ## 573                1    1   7   1      1    44  573
    ## 574                1    0   6   1      2    29  574
    ## 575                0    1   6   3      2     2  575
    ## 576                0    1   6   3      1    41  576
    ## 577                0    0   3   2      1    22  577
    ## 578                1    1   1   3      1    29  578
    ## 579                1    0   7   3      3    45  579
    ## 580                0    0   5   1      2    21  580
    ## 581                1    1   6   3      1    42  581
    ## 582                1    0   5   3      2    10  582
    ## 583                1    0   6   1      3    35  583
    ## 584                1    0   6   3      1    35  584
    ## 585                0    0   6   3      1    11  585
    ## 586                0    1   7   3      2     1  586
    ## 587                1    1   6   1      2    47  587
    ## 588                0    1   7   3      1    14  588
    ## 589                1    0   7   1      1    31  589
    ## 590                0    0   7   3      3     4  590
    ## 591                1    0   7   1      2    23  591
    ## 592                1    0   7   3      1    33  592
    ## 593                1    0   4   1      2    27  593
    ## 594                1    0   4   3      1    35  594
    ## 595                0    1   6   1      1    44  595
    ## 596                1    0   1   3      3     6  596
    ## 597                1    0   7   1      2    38  597
    ## 598                1    0   2   1      1    41  598
    ## 599                1    0   3   1      1    45  599
    ## 600                1    1   7   2      2    39  600
    ## 601                1    1   4   3      1     4  601
    ## 602                1    0   5   1      1    47  602
    ## 603                0    1   7   1      1    28  603
    ## 604                1    0   4   3      2     4  604
    ## 605                0    1   7   2      1     1  605
    ## 606                1    0   6   1      1    34  606
    ## 607                0    0   6   3      3    27  607
    ## 608                0    0   2   1      3    49  608
    ## 609                1    0   4   3      1    44  609
    ## 610                1    0   6   3      2     1  610
    ## 611                0    1   6   3      2     4  611
    ## 612                1    1   7   3      1    36  612
    ## 613                1    0   7   1      2    41  613
    ## 614                1    1   4   1      1    36  614
    ## 615                1    0   3   3      1    31  615
    ## 616                1    1   7   1      1    38  616
    ## 617                1    0   1   1      2    29  617
    ## 618                1    0   6   3      1    15  618
    ## 619                1    0   5   1      1    24  619
    ## 620                0    1   1   3      1    12  620
    ## 621                1    0   1   3      1    28  621
    ## 622                1    0   6   1      1    22  622
    ## 623                1    1   7   3      2    39  623
    ## 624                1    0   6   3      1    15  624
    ## 625                1    0   3   1      1    30  625
    ## 626                0    1   6   2      1    28  626
    ## 627                1    0   7   3      1    29  627
    ## 628                1    0   6   3      2    30  628
    ## 629                1    0   6   2      1    44  629
    ## 630                0    0   3   3      1    22  630
    ## 631                1    1   1   1      3    36  631
    ## 632                1    0   7   3      2    45  632
    ## 633                0    0   4   1      1     6  633
    ## 634                1    0   7   1      2    35  634
    ## 635                1    0   7   3      1    44  635
    ## 636                1    1   4   1      2    35  636
    ## 637                0    0   7   3      1    14  637
    ## 638                1    0   5   1      1    14  638
    ## 639                0    0   1   1      2    28  639
    ## 640                0    0   7   1      2    36  640
    ## 641                1    0   7   2      1    21  641
    ## 642                1    0   5   3      2    42  642
    ## 643                0    0   5   1      1    36  643
    ## 644                1    0   3   3      1    16  644
    ## 645                1    0   7   1      1    47  645
    ## 646                0    0   3   3      2    10  646
    ## 647                0    1   5   3      1     3  647
    ## 648                1    0   6   2      2    30  648
    ## 649                0    0   1   3      1    14  649
    ## 650                0    1   7   1      1     6  650
    ## 651                0    1   6   3      2    42  651
    ## 652                1    1   5   3      1    45  652
    ## 653                0    1   5   1      1    39  653
    ## 654                1    0   4   1      3    42  654
    ## 655                1    1   7   2      1    31  655
    ## 656                1    0   7   2      1    47  656
    ## 657                1    0   4   1      3    20  657
    ## 658                1    1   7   1      3    19  658
    ## 659                1    0   6   2      1    17  659
    ## 660                0    1   3   3      2     1  660
    ## 661                0    0   5   3      2     4  661
    ## 662                1    0   5   3      3    47  662
    ## 663                0    0   3   3      1    39  663
    ## 664                1    0   4   3      1    50  664
    ## 665                1    0   1   3      2    39  665
    ## 666                1    1   4   3      2    10  666
    ## 667                1    0   6   2      3    35  667
    ## 668                1    0   7   3      2    41  668
    ## 669                1    1   4   3      1    10  669
    ## 670                0    1   3   3      1     1  670
    ## 671                0    0   6   3      2    22  671
    ## 672                0    0   4   2      1    22  672
    ## 673                0    0   5   3      1    22  673
    ## 674                1    1   7   3      1    21  674
    ## 675                0    0   4   1      3     6  675
    ## 676                1    0   4   3      1    18  676
    ## 677                0    1   7   1      2    31  677
    ## 678                0    0   7   2      3     7  678
    ## 679                1    0   4   3      3    44  679
    ## 680                1    0   7   3      3    10  680
    ## 681                1    0   3   1      1    12  681
    ## 682                1    0   4   3      2    24  682
    ## 683                1    0   7   3      3    27  683
    ## 684                0    0   4   3      1    22  684
    ## 685                0    0   4   3      1     9  685
    ## 686                0    1   2   3      2    10  686
    ## 687                1    0   1   3      3    28  687
    ## 688                1    1   6   3      3    29  688
    ## 689                1    0   7   3      1    24  689
    ## 690                0    0   6   1      1    24  690
    ## 691                1    1   3   1      1    22  691
    ## 692                1    0   4   3      1    22  692
    ## 693                0    0   1   1      1    12  693
    ## 694                1    0   6   3      1    18  694
    ## 695                0    1   6   1      1    32  695
    ## 696                1    1   6   2      1    47  696
    ## 697                0    1   7   1      3    22  697
    ## 698                0    0   6   3      2    33  698
    ## 699                0    0   4   1      2    10  699
    ## 700                0    0   4   3      2    12  700
    ## 701                1    1   6   1      1    41  701
    ## 702                0    1   6   2      1    20  702
    ## 703                0    1   4   3      1    11  703
    ## 704                1    0   1   3      2     8  704
    ## 705                1    0   6   3      1    21  705
    ## 706                1    0   7   2      2    14  706
    ## 707                1    0   7   3      2    12  707
    ## 708                1    0   7   3      1    30  708
    ## 709                1    0   6   3      3     1  709
    ## 710                0    1   7   2      1    12  710
    ## 711                1    0   1   1      2    12  711
    ## 712                1    1   5   3      2    35  712
    ## 713                1    1   7   3      1     1  713
    ## 714                1    0   6   1      1    39  714
    ## 715                1    1   1   2      1    44  715
    ## 716                0    0   3   2      2    12  716
    ## 717                1    0   5   2      1    47  717
    ## 718                1    0   7   3      1     7  718
    ## 719                1    0   6   3      2    14  719
    ## 720                1    0   5   3      2     4  720
    ## 721                1    0   6   2      1    33  721
    ## 722                1    0   3   3      3    31  722
    ## 723                0    1   6   3      1    30  723
    ## 724                1    0   5   3      1    28  724
    ## 725                1    0   6   3      2    45  725
    ## 726                1    1   7   2      1    47  726
    ## 727                1    0   6   3      1    26  727
    ## 728                0    0   4   3      2    11  728
    ## 729                0    0   4   1      2    20  729
    ## 730                1    0   6   1      2     5  730
    ## 731                1    1   6   1      1    33  731
    ## 732                0    1   7   3      1    12  732
    ## 733                0    1   7   3      1     4  733
    ## 734                1    0   7   3      1    30  734
    ## 735                1    1   3   1      2    42  735
    ## 736                1    1   2   3      3    41  736
    ## 737                1    0   6   3      3    39  737
    ## 738                1    1   6   3      1    29  738
    ## 739                1    0   7   3      3    18  739
    ## 740                1    0   4   3      1     8  740
    ## 741                1    0   7   3      1    11  741
    ## 742                1    0   6   3      2    36  742
    ## 743                0    1   7   1      2    12  743
    ## 744                0    0   6   3      1    30  744
    ## 745                1    0   5   1      3    45  745
    ## 746                1    0   7   3      1    42  746
    ## 747                1    0   5   2      2    30  747
    ## 748                1    0   7   3      1    11  748
    ## 749                0    1   6   2      2    34  749
    ## 750                1    0   4   3      1    30  750
    ## 751                0    0   5   2      1    12  751
    ## 752                1    0   6   3      1    14  752
    ## 753                0    0   5   3      1    41  753
    ## 754                0    1   4   2      1     1  754
    ## 755                0    0   7   1      2    41  755
    ## 756                0    0   6   3      1     1  756
    ## 757                1    0   1   3      1    28  757
    ## 758                0    1   1   3      1    17  758
    ## 759                1    0   7   3      2    10  759
    ## 760                0    0   4   3      1    22  760
    ## 761                1    0   5   3      1    38  761
    ## 762                1    0   7   3      3     4  762
    ## 763                1    0   7   1      3     1  763
    ## 764                1    0   5   1      2    38  764
    ## 765                1    0   4   3      2    20  765
    ## 766                0    0   3   1      1     7  766
    ## 767                1    1   4   3      2    45  767
    ## 768                1    0   4   1      1    12  768
    ## 769                1    0   1   3      1    34  769
    ## 770                1    1   3   1      3    41  770
    ## 771                0    0   4   1      1    21  771
    ## 772                1    0   6   1      2    27  772
    ## 773                1    0   4   1      2    29  773
    ## 774                1    0   7   3      1    24  774
    ## 775                0    0   6   1      2    41  775
    ## 776                1    0   6   1      3    35  776
    ## 777                0    0   1   1      2     7  777
    ## 778                1    0   7   2      3    10  778
    ## 779                1    0   6   1      1     5  779
    ## 780                0    1   4   3      1    28  780
    ## 781                0    1   6   2      2    39  781
    ## 782                0    1   7   1      2     7  782
    ## 783                0    0   3   1      1     5  783
    ## 784                0    1   6   1      2    31  784
    ## 785                1    0   6   2      1     6  785
    ## 786                0    1   6   3      1    11  786
    ## 787                0    1   1   3      1    28  787
    ## 788                1    1   4   2      1    22  788
    ## 789                1    0   4   1      2    11  789
    ## 790                0    1   5   1      1    12  790
    ## 791                1    0   7   3      1    33  791
    ## 792                1    1   1   3      2    34  792
    ## 793                1    0   6   1      1    21  793
    ## 794                0    0   4   3      2    11  794
    ## 795                1    0   7   2      2     5  795
    ## 796                1    0   4   1      1    42  796
    ## 797                0    0   6   2      2    43  797
    ## 798                0    0   7   3      1    33  798
    ## 799                1    0   2   3      3    21  799
    ## 800                0    1   3   2      1     2  800
    ## 801                1    0   7   1      1    27  801
    ## 802                1    0   6   3      2    44  802
    ## 803                1    0   4   1      3     7  803
    ## 804                1    1   3   3      2     6  804
    ## 805                0    0   7   1      1    43  805
    ## 806                1    0   3   1      1     6  806
    ## 807                1    1   7   1      2     6  807
    ## 808                1    0   7   3      2    47  808
    ## 809                0    0   3   2      1    12  809
    ## 810                0    0   5   1      1    23  810
    ## 811                1    0   6   2      1    35  811
    ## 812                1    0   6   3      1    47  812
    ## 813                0    0   7   1      2    41  813
    ## 814                0    0   3   3      1    20  814
    ## 815                0    1   7   1      1    33  815
    ## 816                1    0   5   3      1    44  816
    ## 817                1    0   7   1      2    33  817
    ## 818                0    0   6   3      1    12  818
    ## 819                1    0   3   3      3    28  819
    ## 820                1    0   7   3      3    18  820
    ## 821                1    0   6   3      1    49  821
    ## 822                0    0   4   1      2    41  822
    ## 823                1    0   7   3      2    39  823
    ## 824                1    1   7   1      1    45  824
    ## 825                1    0   6   1      3    38  825
    ## 826                1    0   7   3      2    32  826
    ## 827                0    0   1   3      1    28  827
    ## 828                1    0   3   3      2    39  828
    ## 829                0    1   6   3      1    13  829
    ## 830                0    0   3   1      2    27  830
    ## 831                0    0   7   3      3    22  831
    ## 832                0    0   1   1      3    10  832
    ## 833                1    0   1   3      1    45  833
    ## 834                1    1   4   2      2    39  834
    ## 835                1    0   4   3      2    22  835
    ## 836                0    0   7   3      1    34  836
    ## 837                0    1   2   1      2     3  837
    ## 838                1    0   7   3      1    11  838
    ## 839                1    0   5   3      2    31  839
    ## 840                1    0   7   3      1    14  840
    ## 841                1    0   3   3      1    35  841
    ## 842                1    0   6   3      2    41  842
    ## 843                1    0   7   3      2    36  843
    ## 844                1    0   3   3      2    35  844
    ## 845                1    0   6   1      1    18  845
    ## 846                1    1   4   3      1    29  846
    ## 847                1    1   6   3      2    33  847
    ## 848                0    1   3   1      3    29  848
    ## 849                1    0   4   1      2    34  849
    ## 850                1    0   7   1      2    37  850
    ## 851                1    0   1   3      1     5  851
    ## 852                1    1   7   1      1    15  852
    ## 853                0    1   7   3      1    41  853
    ## 854                1    0   7   1      1    14  854
    ## 855                1    1   3   1      1    39  855
    ## 856                0    1   7   3      2    30  856
    ## 857                1    0   4   1      2    18  857
    ## 858                1    0   7   1      1     3  858
    ## 859                0    0   6   3      1    12  859
    ## 860                1    0   7   3      1    40  860
    ## 861                1    0   6   1      2    10  861
    ## 862                0    0   7   2      1     4  862
    ## 863                1    0   4   2      1    44  863
    ## 864                0    1   6   3      2     4  864
    ## 865                0    0   4   1      1    23  865
    ## 866                1    0   5   3      1    36  866
    ## 867                0    0   5   3      1     9  867
    ## 868                0    0   6   2      2    22  868
    ## 869                1    0   7   1      1    19  869
    ## 870                1    0   6   1      3    21  870
    ## 871                1    0   6   3      1    18  871
    ## 872                1    0   7   1      2    39  872
    ## 873                1    0   6   3      1    28  873
    ## 874                0    1   4   1      1    47  874
    ## 875                0    0   2   3      1    44  875
    ## 876                1    0   5   1      2    45  876
    ## 877                1    0   7   3      1     6  877
    ## 878                1    1   5   3      1     1  878
    ## 879                1    1   6   3      1    41  879
    ## 880                1    0   5   3      1    10  880
    ## 881                1    1   6   3      1    23  881
    ## 882                0    0   5   1      1    31  882
    ## 883                1    0   6   3      1    23  883
    ## 884                1    0   3   3      1    39  884
    ## 885                0    0   6   3      2    12  885
    ## 886                0    1   1   1      2     3  886
    ## 887                1    1   7   1      1    23  887
    ## 888                0    0   1   3      1    49  888
    ## 889                1    0   4   3      1    47  889
    ## 890                1    0   6   3      2    18  890
    ## 891                1    0   7   3      1    11  891
    ## 892                1    0   4   1      3    31  892
    ## 893                0    1   5   2      2     4  893
    ## 894                1    0   6   3      1    41  894
    ## 895                1    0   1   1      2    10  895
    ## 896                1    0   7   1      2    35  896
    ## 897                1    0   6   3      1    22  897
    ## 898                1    1   7   1      2     1  898
    ## 899                1    0   6   1      2     6  899
    ## 900                0    0   6   3      1    12  900
    ## 901                0    1   5   3      2    44  901
    ## 902                1    1   4   3      1    19  902
    ## 903                0    0   2   1      3    21  903
    ## 904                1    0   7   3      1    16  904
    ## 905                0    1   6   3      2    34  905
    ## 906                1    0   6   1      1    31  906
    ## 907                1    0   4   1      1    45  907
    ## 908                1    0   6   3      1    12  908
    ## 909                1    0   1   3      2    35  909
    ## 910                1    0   1   2      2    47  910
    ## 911                1    0   4   3      3    30  911
    ## 912                1    0   5   3      1    33  912
    ## 913                1    0   7   3      1    10  913
    ## 914                1    1   6   3      1    24  914
    ## 915                1    0   6   1      1    38  915
    ## 916                0    1   7   3      2    22  916
    ## 917                1    0   7   3      1    38  917
    ## 918                1    0   5   1      1    44  918
    ## 919                1    0   7   3      1    39  919
    ## 920                1    1   7   3      1    27  920
    ## 921                1    0   7   3      2    33  921
    ## 922                1    0   7   3      1    24  922
    ## 923                0    0   5   1      1    30  923
    ## 924                0    1   5   3      2    39  924
    ## 925                1    0   1   3      3    18  925
    ## 926                0    0   5   1      1     5  926
    ## 927                0    1   2   3      1     6  927
    ## 928                0    0   6   2      2    47  928
    ## 929                1    0   4   3      2    23  929
    ## 930                1    0   4   3      3    43  930
    ## 931                1    0   7   3      2    45  931
    ## 932                1    0   6   2      2    12  932
    ## 933                1    1   1   1      2    29  933
    ## 934                1    0   6   2      3    35  934
    ## 935                1    0   7   1      1    31  935
    ## 936                0    1   2   3      2     9  936
    ## 937                1    0   7   3      2     4  937
    ## 938                1    1   7   3      1    47  938
    ## 939                1    0   1   2      1    14  939
    ## 940                1    0   7   1      2     6  940
    ## 941                1    1   6   3      1    38  941
    ## 942                0    0   1   3      1     2  942
    ## 943                0    0   1   3      3    22  943
    ## 944                1    0   4   2      2     3  944
    ## 945                1    0   6   3      2    44  945
    ## 946                1    0   7   2      2    22  946
    ## 947                1    0   3   2      3    36  947
    ## 948                1    0   7   3      1    41  948
    ## 949                0    0   7   1      1    47  949
    ## 950                1    0   5   3      2     7  950
    ## 951                0    1   7   1      1     6  951
    ## 952                0    0   6   2      2    12  952
    ## 953                1    1   7   3      1    29  953
    ## 954                1    0   3   3      2    35  954
    ## 955                1    0   4   1      3     1  955
    ## 956                0    0   6   3      1     6  956
    ## 957                0    1   1   3      1    44  957
    ## 958                1    0   5   2      1    15  958
    ## 959                1    0   6   1      1    33  959
    ## 960                1    0   6   1      1    35  960
    ## 961                1    1   4   1      2    22  961
    ## 962                1    0   5   3      2    44  962
    ## 963                1    0   4   1      1     6  963
    ## 964                0    0   7   3      1    28  964
    ## 965                0    0   5   1      1     4  965
    ## 966                1    0   3   3      1    27  966
    ## 967                1    0   1   3      3    14  967
    ## 968                1    1   6   3      1    36  968
    ## 969                1    0   7   1      2     1  969
    ## 970                1    0   7   3      1    35  970
    ## 971                1    1   5   2      1    42  971
    ## 972                1    0   6   2      1    30  972
    ## 973                1    0   6   3      1    12  973
    ## 974                0    0   4   3      2    17  974
    ## 975                0    0   7   1      2    31  975
    ## 976                0    0   4   3      1     4  976
    ## 977                0    0   7   1      1    24  977
    ## 978                1    0   5   1      1    47  978
    ## 979                1    1   1   3      2    39  979
    ## 980                1    0   1   3      1     6  980
    ## 981                0    1   4   1      1    23  981
    ## 982                0    1   7   2      2    30  982
    ## 983                1    0   7   1      1     6  983
    ## 984                1    0   4   3      1    11  984
    ## 985                1    1   5   1      2    23  985
    ## 986                1    0   7   3      1    31  986
    ## 987                1    1   4   1      1    29  987
    ## 988                1    0   4   3      1    45  988
    ## 989                1    0   7   1      2     4  989
    ## 990                0    0   3   1      2    45  990
    ## 991                0    0   7   3      2    12  991
    ## 992                1    0   6   3      2     1  992
    ## 993                1    0   6   3      2    39  993
    ## 994                1    0   6   2      1    11  994
    ## 995                1    0   5   1      2    11  995
    ## 996                1    0   7   3      1     6  996
    ## 997                1    0   7   3      2    18  997
    ## 998                0    1   6   3      1    12  998
    ## 999                1    0   4   3      1    31  999
    ## 1000               1    0   5   2      3    44 1000
    ## 1001               0    1   3   1      2    34 1001
    ## 1002               1    0   7   2      1    36 1002
    ## 1003               1    0   5   3      3    35 1003
    ## 1004               1    1   4   3      1    45 1004
    ## 1005               0    1   1   2      1    43 1005
    ## 1006               1    0   3   1      3    44 1006
    ## 1007               0    1   4   1      1    18 1007
    ## 1008               1    0   3   3      2     9 1008
    ## 1009               1    0   6   3      1    31 1009
    ## 1010               1    0   6   3      1    24 1010
    ## 1011               1    0   7   1      3    43 1011
    ## 1012               0    1   7   1      1    12 1012
    ## 1013               1    0   4   1      1    19 1013
    ## 1014               1    1   3   1      1    35 1014
    ## 1015               1    1   4   3      3    39 1015
    ## 1016               1    0   6   3      1    18 1016
    ## 1017               1    0   4   3      2    10 1017
    ## 1018               0    0   1   3      1    28 1018
    ## 1019               1    0   5   2      1    38 1019
    ## 1020               0    1   7   3      2    20 1020
    ## 1021               1    1   7   1      3     6 1021
    ## 1022               1    0   7   3      1    41 1022
    ## 1023               0    0   6   2      1    20 1023
    ## 1024               0    0   7   1      1    41 1024
    ## 1025               1    0   6   2      1    38 1025
    ## 1026               1    0   4   3      2     1 1026
    ## 1027               1    0   7   1      1    34 1027
    ## 1028               1    1   6   3      1    19 1028
    ## 1029               1    1   6   3      1    14 1029
    ## 1030               0    0   6   1      1    12 1030
    ## 1031               0    0   3   3      1    33 1031
    ## 1032               1    0   6   3      1    26 1032
    ## 1033               0    0   4   1      3     4 1033
    ## 1034               0    0   7   1      2    21 1034
    ## 1035               0    1   4   3      1    34 1035
    ## 1036               1    0   7   2      2    47 1036
    ## 1037               1    0   4   2      1    44 1037
    ## 1038               1    0   5   3      2    11 1038
    ## 1039               0    1   7   3      2    41 1039
    ## 1040               1    0   6   3      3    13 1040
    ## 1041               1    0   5   1      2    27 1041
    ## 1042               1    0   6   2      2    42 1042
    ## 1043               1    0   6   1      1    18 1043
    ## 1044               1    1   7   1      3    22 1044
    ## 1045               0    0   4   2      1     4 1045
    ## 1046               1    0   7   1      2    31 1046
    ## 1047               0    0   4   1      2    26 1047
    ## 1048               1    0   6   3      1    29 1048
    ## 1049               1    0   4   3      1    22 1049
    ## 1050               0    1   4   3      1    45 1050
    ## 1051               1    0   7   3      2    49 1051
    ## 1052               1    0   4   3      2    38 1052
    ## 1053               0    0   3   3      2    48 1053
    ## 1054               1    0   7   1      1    18 1054
    ## 1055               1    0   6   1      2     6 1055
    ## 1056               0    0   7   3      2    34 1056
    ## 1057               1    1   7   3      1    47 1057
    ## 1058               1    0   1   3      2    10 1058
    ## 1059               1    0   6   2      3    47 1059
    ## 1060               1    0   4   1      1     1 1060
    ## 1061               0    1   7   1      1    31 1061
    ## 1062               0    0   1   2      1    39 1062
    ## 1063               1    0   5   1      2    29 1063
    ## 1064               1    1   7   2      3    36 1064
    ## 1065               0    0   1   3      1    21 1065
    ## 1066               1    0   6   1      1    21 1066
    ## 1067               1    0   6   3      1     5 1067
    ## 1068               1    1   6   3      1     5 1068
    ## 1069               1    0   4   1      3     7 1069
    ## 1070               1    0   5   3      1    34 1070
    ## 1071               1    0   5   3      2    19 1071
    ## 1072               1    1   7   3      3    38 1072
    ## 1073               1    0   5   3      2     6 1073
    ## 1074               1    0   6   3      2     1 1074
    ## 1075               0    0   3   1      1     7 1075
    ## 1076               1    0   7   1      2    38 1076
    ## 1077               1    0   1   1      2    36 1077
    ## 1078               0    1   7   1      1    22 1078
    ## 1079               0    1   7   1      1    14 1079
    ## 1080               1    0   1   2      2    45 1080
    ## 1081               1    0   6   3      3     1 1081
    ## 1082               1    0   1   1      1    10 1082
    ## 1083               0    0   6   3      2     3 1083
    ## 1084               1    0   7   3      1     5 1084
    ## 1085               1    0   6   1      1    41 1085
    ## 1086               0    1   7   3      1    21 1086
    ## 1087               1    0   5   3      2     1 1087
    ## 1088               1    0   7   3      1    34 1088
    ## 1089               1    0   7   1      2    38 1089
    ## 1090               0    0   6   2      2     1 1090
    ## 1091               1    0   1   3      1    30 1091
    ## 1092               1    0   6   3      1    30 1092
    ## 1093               1    0   6   1      1    35 1093
    ## 1094               1    0   7   3      1    38 1094
    ## 1095               1    0   5   3      1     6 1095
    ## 1096               1    0   6   2      2     1 1096
    ## 1097               1    0   7   3      2    41 1097
    ## 1098               1    1   3   1      2    47 1098
    ## 1099               1    0   7   3      1     1 1099
    ## 1100               1    0   7   3      2    36 1100
    ## 1101               0    0   6   1      1    31 1101
    ## 1102               1    0   7   1      1    41 1102
    ## 1103               0    0   7   3      2    41 1103
    ## 1104               1    0   5   3      1     3 1104
    ## 1105               1    0   7   1      1    45 1105
    ## 1106               0    1   3   1      1    34 1106
    ## 1107               1    1   6   1      1    35 1107
    ## 1108               1    0   5   3      1     3 1108
    ## 1109               1    0   4   3      2    39 1109
    ## 1110               0    1   4   3      1    43 1110
    ## 1111               1    0   7   1      1    35 1111
    ## 1112               1    0   4   3      1     6 1112
    ## 1113               0    0   3   3      3    24 1113
    ## 1114               1    1   6   1      2    45 1114
    ## 1115               0    0   5   1      3     1 1115
    ## 1116               1    1   7   1      3    35 1116
    ## 1117               1    0   4   3      1    32 1117
    ## 1118               1    0   7   2      1    47 1118
    ## 1119               1    1   1   3      3    14 1119
    ## 1120               0    0   4   1      2    28 1120
    ## 1121               1    1   6   3      1    31 1121
    ## 1122               1    0   5   3      3    36 1122
    ## 1123               1    0   5   3      1    29 1123
    ## 1124               1    0   1   3      2     4 1124
    ## 1125               1    0   6   3      3    11 1125
    ## 1126               0    0   6   3      1     6 1126
    ## 1127               1    0   6   1      1    14 1127
    ## 1128               1    0   7   1      1    41 1128
    ## 1129               1    0   6   3      3     6 1129
    ## 1130               0    1   1   3      3    23 1130
    ## 1131               0    0   4   3      1    41 1131
    ## 1132               1    0   1   3      2    35 1132
    ## 1133               1    0   4   2      2    11 1133
    ## 1134               0    0   1   3      1    36 1134
    ## 1135               0    0   4   3      1    42 1135
    ## 1136               1    0   6   1      1    28 1136
    ## 1137               1    0   7   3      2    24 1137
    ## 1138               1    0   3   2      1    28 1138
    ## 1139               1    0   5   3      2    41 1139
    ## 1140               1    0   3   1      1    21 1140
    ## 1141               1    0   6   3      1    33 1141
    ## 1142               1    0   7   2      2    41 1142
    ## 1143               1    0   7   3      2    12 1143
    ## 1144               1    0   7   1      2    35 1144
    ## 1145               1    0   4   3      2    22 1145
    ## 1146               0    0   6   1      2    23 1146
    ## 1147               1    0   6   1      2    10 1147
    ## 1148               1    0   6   3      1     6 1148
    ## 1149               1    0   7   3      3     1 1149
    ## 1150               0    1   7   3      2    17 1150
    ## 1151               1    0   6   3      2    47 1151
    ## 1152               0    0   7   1      2    45 1152
    ## 1153               1    0   5   3      3     1 1153
    ## 1154               1    0   1   3      1    47 1154
    ## 1155               1    1   7   3      1    29 1155
    ## 1156               0    0   4   1      2    39 1156
    ## 1157               0    1   5   1      2    22 1157
    ## 1158               0    1   5   2      3     1 1158
    ## 1159               1    0   4   1      1    14 1159
    ## 1160               1    0   7   3      1    22 1160
    ## 1161               1    0   7   1      1    14 1161
    ## 1162               0    1   3   3      2    41 1162
    ## 1163               1    0   6   3      1    14 1163
    ## 1164               1    0   6   3      1    34 1164
    ## 1165               0    0   6   1      1     6 1165
    ## 1166               0    0   5   1      2    49 1166
    ## 1167               1    0   7   3      1    21 1167
    ## 1168               1    0   7   1      1     1 1168
    ## 1169               1    0   4   1      1    36 1169
    ## 1170               0    1   5   1      1     4 1170
    ## 1171               1    1   6   3      1    34 1171
    ## 1172               0    0   5   3      1    12 1172
    ## 1173               1    0   5   3      1     1 1173
    ## 1174               1    0   7   1      1    45 1174
    ## 1175               1    0   2   3      1     6 1175
    ## 1176               1    0   1   1      2    47 1176
    ## 1177               0    0   5   3      1     4 1177
    ## 1178               0    0   6   1      2     4 1178
    ## 1179               1    0   6   2      1    41 1179
    ## 1180               1    1   4   3      2    14 1180
    ## 1181               1    0   7   3      3    44 1181
    ## 1182               1    0   1   1      1    36 1182
    ## 1183               1    0   6   3      2    33 1183
    ## 1184               0    1   4   1      2    22 1184
    ## 1185               1    0   4   1      1    41 1185
    ## 1186               1    0   7   2      3    44 1186
    ## 1187               1    0   7   3      1    38 1187
    ## 1188               1    0   7   1      3    47 1188
    ## 1189               0    0   1   3      2    12 1189
    ## 1190               1    0   6   1      3    34 1190
    ## 1191               1    1   4   3      1    18 1191
    ## 1192               1    0   6   1      2    39 1192
    ## 1193               1    0   5   2      1     3 1193
    ## 1194               1    0   1   1      2    34 1194
    ## 1195               1    1   5   3      1    33 1195
    ## 1196               1    0   4   1      1    22 1196
    ## 1197               0    0   6   3      1    30 1197
    ## 1198               1    0   3   3      1    22 1198
    ## 1199               1    0   4   2      1    10 1199
    ## 1200               0    0   1   1      2    17 1200

Now that we have our simulated data, we now need our hierarchal stan
model.

``` 

// Index values, observations, and covariates.
data {
  int<lower = 1> N;                      // Number of observations.
  int<lower = 1> K;                      // Number of groups.
  int<lower = 1> I;                      // Number of observation-level covariates.
  
  real satisfaction[N];                        // Vector of observations.
  int<lower = 1, upper = K> male[N];        // Vector of group assignments.
  int eth[N];                               // Vector of ethnicity covariates.
  int age[N];                               // Vector of age covariates.
  int income[N];                          // Vector of income covariates.
  int state[N];                          // Vector of state covariates.
  
  real gamma_mean;                       // Mean for the hyperprior on gamma.
  real<lower = 0> gamma_var;             // Variance for the hyperprior on gamma.
  real<lower = 0> tau_min;               // Minimum for the hyperprior on tau.
  real<lower = 0> tau_max;               // Maximum for the hyperprior on tau.
  real<lower = 0> sigma_min;             // Minimum for the hyperprior on tau.
  real<lower = 0> sigma_max;             // Maximum for the hyperprior on tau.
}

// Parameters and hyperparameters.
parameters {
  matrix[K, (I - 1)] alpha;              // Matrix of observation-level brand coefficients.
  vector[K] beta;                        // Vector of observation-level price coefficients.
  real gamma;                            // Mean of the population model.
  real<lower=0> tau;                     // Variance of the population model.
  real<lower=0> sigma;                   // Variance of the observation model.
}

// Hierarchical regression.
model {
  // Declare mu for use in the linear model.
  vector[N] mu;
  
  // Hyperpriors and prior.
  gamma ~ normal(gamma_mean, gamma_var);
  tau ~ uniform(tau_min, tau_max);
  sigma ~ uniform(sigma_min, sigma_max);

  // Population model and likelihood.
  for (k in 1:K) {
    alpha[k,] ~ normal(gamma, tau);
    beta[k] ~ normal(gamma, tau);
  }
  for (n in 1:N) {
    mu[n] = alpha[male[n], eth[n]] + beta[male[n]] * age[n] + beta[male[n]] * 
    income[n] + beta[male[n]] * state[n];
  }
  cat_pref ~ normal(mu, sigma);
}

// Generate predictions using the posterior.
generated quantities {
  vector[N] mu_pc;                       // Declare mu for predicted linear model.
  real cat_pref_pc[N];                     // Vector of predicted observations.

  // Generate posterior prediction distribution.
  for (n in 1:N) {
    mu_pc[n] = alpha[male[n], eth[n]] + beta[male[n]] * age[n] + beta[male[n]] * 
    income[n] + beta[male[n]] * state[n];
    cat_pref_pc[n] = normal_rng(mu_pc[n], sigma);
  }
}
```

As we will also need to do some prior and posterior checks, and for that
we will need the generative model.

``` 

// Index values, observations, covariates, and hyperior values.
data {
  int<lower = 1> N;                      // Number of observations.
  int<lower = 1> K;                      // Number of groups.
  int<lower = 1> I;                      // Number of observation-level covariates.
  
  int<lower = 1, upper = K> male[N];        // Vector of group assignments.
  int eth[N];                               // Vector of ethnicity covariates.
  int age[N];                               // Vector of age covariates.
  int income[N];                          // Vector of income covariates.
  int state[N];                          // Vector of state covariates.
  
  real gamma_mean;                       // Mean for the hyperprior on gamma.
  real<lower = 0> gamma_var;             // Variance for the hyperprior on gamma.
  real<lower = 0> tau_min;               // Minimum for the hyperprior on tau.
  real<lower = 0> tau_max;               // Maximum for the hyperprior on tau.
  real<lower = 0> sigma_min;             // Minimum for the hyperprior on tau.
  real<lower = 0> sigma_max;             // Maximum for the hyperprior on tau.
}

// Generate data according to the hierarchical regression.
generated quantities {
  matrix[K, (I - 1)] alpha;              // Matrix of observation-level brand coefficients.
  vector[K] beta;                        // Vector of observation-level price coefficients.
  real gamma;                            // Mean of the population model.
  real<lower=0> tau;                     // Variance of the population model.
  real<lower=0> sigma;                   // Variance of the observation model.
  
  vector[N] mu;                          // Declare mu for linear model.
  real cat_pref[N];                        // Vector of observations.

  gamma = normal_rng(gamma_mean, gamma_var);
  tau = uniform_rng(tau_min, tau_max);
  sigma = uniform_rng(sigma_min, sigma_max);

  // Draw parameter values and generate data.
  for (k in 1:K) {
    for (i in 1:(I - 1)) {
      alpha[k, i] = normal_rng(gamma, tau);
    }
    beta[k] = normal_rng(gamma, tau);
  }
  for (n in 1:N) {
    mu[n] = alpha[male[n], eth[n]] + beta[male[n]] * age[n] + beta[male[n]] * 
    income[n] + beta[male[n]] * state[n];
    cat_pref[n] = normal_rng(mu[n], sigma);
  }
}
```

``` r
sim_values <- list(
  N = 6300,                                       # Number of observations.
  K = 2,                                         # Number of groups.
  I = 65,                                         # Number of observation-level covariates.
  
  male = sample(2, 6300, replace = TRUE),            # Vector of group assignments.
  eth = sample(3, 6300, replace = TRUE),       # Vector of brands covariates.
  age = sample(7, 6300, replace = TRUE),        # Vector of price covariates.
  income = sample(3, 6300, replace = TRUE),
  state = sample(50, 6300, replace = TRUE),
  
  gamma_mean = 0.5,                              # Mean for the hyperprior on gamma.
  gamma_var = 0.2,                               # Variance for the hyperprior on gamma.
  tau_min = 0,                                   # Minimum for the hyperprior on tau.
  tau_max = .1,                                   # Maximum for the hyperprior on tau.
  sigma_min = 0,                                 # Minimum for the hyperprior on tau.
  sigma_max = .1                                  # Maximum for the hyperprior on tau.
)
# Generate data.
sim_data <- stan(
  file = here::here("Code/mrp/generate_data.stan"),
  data = sim_values,
  iter = 10,
  chains = 1,
  seed = 42,
  algorithm = "Fixed_param"
)
```

    ## 
    ## SAMPLING FOR MODEL 'generate_data' NOW (CHAIN 1).
    ## Chain 1: Iteration: 1 / 10 [ 10%]  (Sampling)
    ## Chain 1: Iteration: 2 / 10 [ 20%]  (Sampling)
    ## Chain 1: Iteration: 3 / 10 [ 30%]  (Sampling)
    ## Chain 1: Iteration: 4 / 10 [ 40%]  (Sampling)
    ## Chain 1: Iteration: 5 / 10 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 6 / 10 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 7 / 10 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 8 / 10 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 9 / 10 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 10 / 10 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 0 seconds (Warm-up)
    ## Chain 1:                0.017 seconds (Sampling)
    ## Chain 1:                0.017 seconds (Total)
    ## Chain 1:

``` r
# Extract the simulated data.
prior_pc <- tibble(
  cat_pref = as.vector(extract(sim_data)$cat_pref)
)
# Plot the prior predictive distribution.
prior_pc %>% 
  ggplot(aes(x = cat_pref)) +
  geom_density()
```

![](Case-Study-_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
# # Specify data.
# data <- list(
#   N = nrow(true_popn),                     # Number of observations.
#   K = 2,                    # Number of groups.
#   I = 65,                                # Number of observation-level covariates.
#   
#   cat_pref = true_popn$cat_pref,               # Vector of observations.
#   male = as.numeric(as.factor(true_popn$male)),            # Vector of group assignments.
#   eth = true_popn$eth,       # Vector of brands covariates.
#   age = true_popn$age,        # Vector of price covariates.
#   income = true_popn$income,
#   state = true_popn$state,
#   
#   gamma_mean = 0.5,                                # Mean for the hyperprior on gamma.
#   gamma_var = .2,                                 # Variance for the hyperprior on gamma.
#   tau_min = 0,                                   # Minimum for the hyperprior on tau.
#   tau_max = 1,                                   # Maximum for the hyperprior on tau.
#   sigma_min = 0,                                 # Minimum for the hyperprior on tau.
#   sigma_max = 1                                 # Maximum for the hyperprior on tau.
# )
# # Calibrate the model.
# model03 <- stan(
#   file = here::here("Code/mrp/hierarchical_model.stan"),
#   data = data,
#   control = list(adapt_delta = 0.99),
#   seed = 42
# )
```

``` r
# Posterior predictive check.
# post_pc03 <- tibble(
#   # Extract the posterior predicted values.
#   intent = as.vector(extract(model03)$cat_pref_pc)
# )
# # Plot the posterior predictive distribution.
# ggplot(post_pc03, aes(x = intent)) +
#   geom_histogram() +
#   xlim(1, 10) 
# ggplot(purchase_intent, aes(x = intent)) +
#   geom_histogram() +
#   xlim(1, 10) 
```

``` r
# model03 %>%
#   gather_draws(alpha[n, i]) %>%
#   unite(.variable, .variable, n, i) %>%
#   ggplot(aes(x = .value, y = .variable)) +
#   geom_halfeyeh(.width = .95) +
#   facet_wrap(
#     ~ .variable,
#     nrow = data$K,
#     ncol = (data$I - 1),
#     scales = "free"
#   )
```

``` r
# Plot the betas.
# model03 %>%
#   gather_draws(beta[i]) %>%
#   unite(.variable, .variable, i) %>%
#   ggplot(aes(x = .value, y = .variable)) +
#   geom_halfeyeh(.width = .95) +
#   facet_wrap(
#     ~ .variable,
#     nrow = data$K,
#     ncol = 1,
#     scales = "free"
#   )
```

### Postratification

Note: for now I have set these chuncks to eval = false because we don t
have a poststrat\_prob yet so it freaks out

For the postratification portion we take the estimate that we get from
the model times poststrat\(/N / sum(postsrat\)N)

``` r
poststrat_prob <- posterior_prob %*% poststrat$N / sum(poststrat$N)
model_popn_pref <- c(mean = mean(poststrat_prob), sd = sd(poststrat_prob))
round(model_popn_pref, 3)
```

Because we are using simulated data we can see how our poststratified
predictions compare to the true population

``` r
sample_popn_pref <- mean(sample$satisfaction)
round(sample_popn_pref, 3)

compare2 <- compare2 +
  geom_hline(yintercept = model_popn_pref[1], colour = '#2ca25f', size = 1) +
  geom_text(aes(x = 5.2, y = model_popn_pref[1] + .025), label = "MRP", colour = '#2ca25f')
bayesplot_grid(compare, compare2,
               grid_args = list(nrow = 1, widths = c(8, 1)))
```
