# Mabigat
Parameter optimizer in the generation of NNUE net using Optuna framework.

## Setup
* Install python version 3.8 or more
* Install optuna
  * pip install optuna
* Intall pandas  
  * pip install pandas
* Download this repo (will upload the whole pack this weekend)

This is tested to work in windows 10.

## Optimization Process

### A. Generate training positions
There are many ways to generate the training positions, you can vary some parameters that include the depth, the number of training positions, the random multipv number and [many others](https://github.com/nodchip/Stockfish/blob/master/docs/gensfen.md).

### B. Generate validation positions
Similar to section A but with higher depth and lower number of positions.

### C. Learning
Once the training and validation positions are generated learning can then proceed. Learning has also a lot of parameters to optimize, this includes the learning rate, max_grad, validation count and [many others](https://github.com/nodchip/Stockfish/blob/master/docs/learn.md).

After learning we will get a NNUE net which is then can be used by Stockfish and other engines that supports NNUE net.

### D. Optuna framework optimizer
Basically we generate multiple NNUE net one at a time varying the parameters in the training, validation and learning processes. Each NNUE net is then tested in an engine vs engine match. For example we want to vary the following training/validation parameters:  
* random_move_count  
* random_multi_pv  
* random_multi_pv_diff  

And the following learning parameters:  
* max_grad  
* lambda1  

##### Code
```python
random_move_count = trial.suggest_int("random_move_count", 5, 20, 1)  # min, high, step
random_multi_pv = trial.suggest_int("random_multi_pv", 4, 8, 1)
random_multi_pv_diff = trial.suggest_int("random_multi_pv_diff", 50, 400, 25)
max_grad = trial.suggest_categorical("max_grad", [0.3, 0.5, 0.75, 1.0])
lambda1 = trial.suggest_categorical("lambda1", [0.0, 0.25, 0.5, 0.75, 1.0])
```

At trial 0 we ask the optimizer for the parameter values to try then we generate the training/validation positions and do the learning, we then save the net as 0_nn.bin, then for trial 1 we follow the same and save 1_nn.bin. Since we now have 2 nets we then create an engine vs engine test for 0_nn.bin and 1_nn.bin in a 100-game match. Once done we report the result to the optuna optimizer with the score from the point of view of the latest net which in this case is trial 1. Then we ask another parameters to try for trial 2 and then test 0_nn.bin against 2_nn.bin. The optimizer will try to give us the parameter values that will defeat 0_nn.bin on a larger margin.

It would look something like this.
```
A new study created in RDB with name: nnue_study_2
Mabigat NNUE parameter optimizer

engine   : nodechip_stockfish_2021-02-08.exe
threads  : 6
hash     : 1024

study name: nnue_study_2
number of trials: 20
optimizer: Optuna with TPE sampler

number of training positions: 2000000
number of validation positions: 2000

starting trial: 0
param to try: {'random_move_count': 9, 'random_multi_pv': 6, 'random_multi_pv_diff': 225, 'max_grad': 0.75, 'lambda1': 0.75}
 number  value  params_lambda1  params_max_grad  params_random_move_count  params_random_multi_pv  params_random_multi_pv_diff     state
      0    0.5            0.75             0.75                         9                       6                          225  COMPLETE

starting trial: 1
param to try: {'random_move_count': 17, 'random_multi_pv': 6, 'random_multi_pv_diff': 350, 'max_grad': 0.3, 'lambda1': 1.0}
 number  value  params_lambda1  params_max_grad  params_random_move_count  params_random_multi_pv  params_random_multi_pv_diff     state
      0  0.500            0.75             0.75                         9                       6                          225  COMPLETE
      1  0.575            1.00             0.30                        17                       6                          350  COMPLETE

starting trial: 2
param to try: {'random_move_count': 9, 'random_multi_pv': 7, 'random_multi_pv_diff': 375, 'max_grad': 0.3, 'lambda1': 0.75}
 number  value  params_lambda1  params_max_grad  params_random_move_count  params_random_multi_pv  params_random_multi_pv_diff     state
      0  0.500            0.75             0.75                         9                       6                          225  COMPLETE
      1  0.575            1.00             0.30                        17                       6                          350  COMPLETE
      2  0.510            0.75             0.30                         9                       7                          375  COMPLETE
```
The column `number` is the trial number, `value` is the result of the engine vs engine match from the point of view of the trial number against the trial numbered 0. Parameters to be optimized follow and the state of the tuning. If value is high that is good for the given parameters.

As the number of trials increases, the optimizer is suggesting good parameters. Trial 12 is doing good against trial 0 at 68.5%.

```
...

starting trial: 15
param to try: {'random_move_count': 17, 'random_multi_pv': 5, 'random_multi_pv_diff': 200, 'max_grad': 0.3, 'lambda1': 1.0}
 number  value  params_lambda1  params_max_grad  params_random_move_count  params_random_multi_pv  params_random_multi_pv_diff     state
      0  0.500            0.75             0.75                         9                       6                          225  COMPLETE
      1  0.575            1.00             0.30                        17                       6                          350  COMPLETE
      2  0.510            0.75             0.30                         9                       7                          375  COMPLETE
      3  0.025            0.75             1.00                         6                       7                          400  COMPLETE
      4  0.060            0.25             0.50                        11                       6                          275  COMPLETE
      5  0.010            0.75             1.00                        10                       7                          275  COMPLETE
      6  0.055            0.00             0.30                        18                       8                           75  COMPLETE
      7  0.120            0.25             0.50                        11                       8                          400  COMPLETE
      8  0.580            1.00             0.30                        13                       5                          300  COMPLETE
      9  0.600            1.00             0.30                        15                       6                          275  COMPLETE
     10  0.230            0.50             0.75                        20                       4                          100  COMPLETE
     11  0.495            1.00             0.30                        15                       4                          300  COMPLETE
     12  0.685            1.00             0.30                        14                       5                          175  COMPLETE
     13  0.555            1.00             0.30                        15                       5                          150  COMPLETE
     14  0.580            1.00             0.30                        14                       5                          175  COMPLETE
     15  0.625            1.00             0.30                        17                       5                          200  COMPLETE
```

## Optimization test

### A. Trial 0 as fix opponent of optimizer
If the net from trial 0 is not good, it would be difficult to know which parameters are really optimized.

##### Parameters to be optimized
```python
random_move_count = trial.suggest_int('random_move_count', 5, 20, 1)
random_multi_pv = trial.suggest_int('random_multi_pv', 4, 8, 1)
random_multi_pv_diff = trial.suggest_int('random_multi_pv_diff', 100, 400, 25)
max_grad = trial.suggest_categorical('max_grad', [0.1, 0.2, 0.3, 0.5, 0.6])
lambda1 = trial.suggest_categorical('lambda1', [0.0, 0.25, 0.5, 0.75, 1.0])
random_move_minply = trial.suggest_int('random_move_minply', 1, 10, 1)
write_minply = trial.suggest_int('write_minply', 6, 26, 1)
skip_dupes_in_training = trial.suggest_categorical('skip_dupes_in_training', [0, 1])
fen_skip_train = trial.suggest_categorical('fen_skip_train', [0, 1])
fen_skip_val = trial.suggest_categorical('fen_skip_val', [0, 1])
```

```
A new study created in RDB with name: nnue_study_3
Mabigat NNUE parameter optimizer

engine   : nodchip_stockfish_2021-02-08.exe
threads  : 6
hash     : 1024

study name: nnue_study_3
number of trials: 20
optimizer: Optuna with TPE sampler

number of training positions: 2000000
number of validation positions: 2000
training depth: 1
validation depth: 4
book: noob_3moves.epd

eval_save_interval: 500000
loss_output_interval: 50000

starting trial: 0
param to try: {'random_move_count': 19, 'random_multi_pv': 5, 'random_multi_pv_diff': 300, 'max_grad': 0.6, 'lambda1': 1.0, 'random_move_minply': 4, 'write_minply': 9, 'skip_dupes_in_training': 0, 'fen_skip_train': 1, 'fen_skip_val': 0}
 number  value  params_fen_skip_train  params_fen_skip_val  params_lambda1  params_max_grad  params_random_move_count  params_random_move_minply  params_random_multi_pv  params_random_multi_pv_diff  params_skip_dupes_in_training  params_write_minply     state
      0    0.5                      1                    0             1.0              0.6                        19                          4                       5                          300                              0                    9  COMPLETE

starting trial: 1
param to try: {'random_move_count': 7, 'random_multi_pv': 6, 'random_multi_pv_diff': 125, 'max_grad': 0.6, 'lambda1': 0.0, 'random_move_minply': 3, 'write_minply': 19, 'skip_dupes_in_training': 0, 'fen_skip_train': 1, 'fen_skip_val': 0}
 number  value  params_fen_skip_train  params_fen_skip_val  params_lambda1  params_max_grad  params_random_move_count  params_random_move_minply  params_random_multi_pv  params_random_multi_pv_diff  params_skip_dupes_in_training  params_write_minply     state
      0  0.500                      1                    0             1.0              0.6                        19                          4                       5                          300                              0                    9  COMPLETE
      1  0.765                      1                    0             0.0              0.6                         7                          3                       6                          125                              0                   19  COMPLETE

...
```

After trial 20 it is a mess, the net in trial 0 is getting punished.

```
starting trial: 19
param to try: {'random_move_count': 5, 'random_multi_pv': 5, 'random_multi_pv_diff': 350, 'max_grad': 0.5, 'lambda1': 0.75, 'random_move_minply': 6, 'write_minply': 26, 'skip_dupes_in_training': 1, 'fen_skip_train': 0, 'fen_skip_val': 0}
 number  value  params_fen_skip_train  params_fen_skip_val  params_lambda1  params_max_grad  params_random_move_count  params_random_move_minply  params_random_multi_pv  params_random_multi_pv_diff  params_skip_dupes_in_training  params_write_minply     state
      0  0.500                      1                    0            1.00              0.6                        19                          4                       5                          300                              0                    9  COMPLETE
      1  0.765                      1                    0            0.00              0.6                         7                          3                       6                          125                              0                   19  COMPLETE
      2  0.985                      1                    0            0.50              0.3                         9                          1                       7                          375                              1                   22  COMPLETE
      3  0.930                      0                    1            0.25              0.2                         7                          3                       6                          125                              1                    7  COMPLETE
      4  0.990                      0                    0            0.75              0.1                        19                         10                       6                          225                              1                    6  COMPLETE
      5  0.845                      0                    0            0.25              0.6                        12                          4                       7                          125                              0                    9  COMPLETE
      6  0.985                      0                    0            1.00              0.5                        10                          4                       8                          275                              0                   11  COMPLETE
      7  0.990                      0                    1            1.00              0.5                        15                         10                       6                          400                              1                    7  COMPLETE
      8  1.000                      0                    0            0.75              0.2                         5                         10                       6                          325                              1                   10  COMPLETE
      9  0.945                      0                    1            0.25              0.2                        19                          4                       6                          375                              1                   10  COMPLETE
     10  1.000                      0                    0            0.75              0.2                        15                          8                       4                          200                              1                   14  COMPLETE
     11  0.990                      0                    0            0.75              0.2                        15                          8                       4                          200                              1                   15  COMPLETE
     12  0.990                      0                    0            0.75              0.2                        15                          8                       4                          325                              1                   14  COMPLETE
     13  0.985                      0                    0            0.75              0.2                         5                          8                       5                          175                              1                   13  COMPLETE
     14  0.990                      0                    0            0.75              0.2                        13                          7                       5                          250                              1                   18  COMPLETE
     15  0.995                      0                    0            0.75              0.3                        17                         10                       7                          325                              1                   17  COMPLETE
     16  0.735                      1                    0            0.00              0.1                        13                          6                       4                          175                              1                   13  COMPLETE
     17  0.990                      0                    1            0.50              0.2                        11                          9                       8                          275                              1                   21  COMPLETE
     18  1.000                      0                    0            0.75              0.2                         5                          6                       5                          350                              1                   26  COMPLETE
     19  0.995                      0                    0            0.75              0.5                         5                          6                       5                          350                              1                   26  COMPLETE

```

Two solutions, first use a good pre-trained nn as fix opponent of optimizer param values or second use the best param found so far as the opponent of the optimizer param values.

### B. Best trial found so far as opponent of optimizer
Whenever a new net defeats the old best net, this new net will become the best net and be the opponent of new net from the parameters suggested by the optimizer.

##### Parameters to be optimized
This is the same parameters used in section A above.

```python
random_move_count = trial.suggest_int('random_move_count', 5, 20, 1)
random_multi_pv = trial.suggest_int('random_multi_pv', 4, 8, 1)
random_multi_pv_diff = trial.suggest_int('random_multi_pv_diff', 100, 400, 25)
max_grad = trial.suggest_categorical('max_grad', [0.1, 0.2, 0.3, 0.5, 0.6])
lambda1 = trial.suggest_categorical('lambda1', [0.0, 0.25, 0.5, 0.75, 1.0])
random_move_minply = trial.suggest_int('random_move_minply', 1, 10, 1)
write_minply = trial.suggest_int('write_minply', 6, 26, 1)
skip_dupes_in_training = trial.suggest_categorical('skip_dupes_in_training', [0, 1])
fen_skip_train = trial.suggest_categorical('fen_skip_train', [0, 1])
fen_skip_val = trial.suggest_categorical('fen_skip_val', [0, 1])
```

```
A new study created in RDB with name: nnue_study_4
Mabigat NNUE parameter optimizer

engine   : ./engine/nodchip_stockfish_2021-03-23.exe
threads  : 6
hash     : 1024

study name: nnue_study_4
number of trials: 50
optimizer: Optuna with TPE sampler

number of training positions: 2000000
number of validation positions: 2000
training depth: 1
validation depth: 4
book: ./opening/noob_3moves.epd

eval_save_interval: 500000
loss_output_interval: 50000

starting trial: 0
param to try: {'random_move_count': 5, 'random_multi_pv': 4, 'random_multi_pv_diff': 150, 'max_grad': 0.2, 'lambda1': 0.5, 'random_move_minply': 6, 'write_minply': 6, 'skip_dupes_in_training': 1, 'fen_skip_train': 0, 'fen_skip_val': 1}
 number  value  params_fen_skip_train  params_fen_skip_val  params_lambda1  params_max_grad  params_random_move_count  params_random_move_minply  params_random_multi_pv  params_random_multi_pv_diff  params_skip_dupes_in_training  params_write_minply    state
      0    0.5                      0                    1             0.5              0.2                         5                          6                       4                          150                              1                    6 COMPLETE

starting trial: 1
param to try: {'random_move_count': 11, 'random_multi_pv': 4, 'random_multi_pv_diff': 275, 'max_grad': 0.5, 'lambda1': 1.0, 'random_move_minply': 3, 'write_minply': 9, 'skip_dupes_in_training': 1, 'fen_skip_train': 1, 'fen_skip_val': 0}
Execute engine vs engine match between 1_nn.bin and 0_nn.bin
Match done! actual result: 0.69, point of view: name=1_nn
use_best_param: True, adjusted result: 0.69
 number  value  params_fen_skip_train  params_fen_skip_val  params_lambda1  params_max_grad  params_random_move_count  params_random_move_minply  params_random_multi_pv  params_random_multi_pv_diff  params_skip_dupes_in_training  params_write_minply    state
      0   0.50                      0                    1             0.5              0.2                         5                          6                       4                          150                              1                    6 COMPLETE
      1   0.69                      1                    0             1.0              0.5                        11                          3                       4                          275                              1                    9 COMPLETE

starting trial: 2
param to try: {'random_move_count': 16, 'random_multi_pv': 4, 'random_multi_pv_diff': 150, 'max_grad': 0.2, 'lambda1': 0.75, 'random_move_minply': 1, 'write_minply': 13, 'skip_dupes_in_training': 1, 'fen_skip_train': 0, 'fen_skip_val': 1}
Execute engine vs engine match between 2_nn.bin and 1_nn.bin
Match done! actual result: 0.833, point of view: name=2_nn
use_best_param: True, adjusted result: 1.023
 number  value  params_fen_skip_train  params_fen_skip_val  params_lambda1  params_max_grad  params_random_move_count  params_random_move_minply  params_random_multi_pv  params_random_multi_pv_diff  params_skip_dupes_in_training  params_write_minply    state
      0  0.500                      0                    1            0.50              0.2                         5                          6                       4                          150                              1                    6 COMPLETE
      1  0.690                      1                    0            1.00              0.5                        11                          3                       4                          275                              1                    9 COMPLETE
      2  1.023                      0                    1            0.75              0.2                        16                          1                       4                          150                              1                   13 COMPLETE

```

After 3 trials net 1 defeats net 0 and net 2 defeates net 1.

Result after trial 36.

```
starting trial: 36
param to try: {'random_move_count': 11, 'random_multi_pv': 4, 'random_multi_pv_diff': 175, 'max_grad': 0.5, 'lambda1': 1.0, 'random_move_minply': 5, 'write_minply': 15, 'skip_dupes_in_training': 1, 'fen_skip_train': 1, 'fen_skip_val': 1}
Execute engine vs engine match between 36_nn.bin and 34_nn.bin
Match done! actual result: 0.555, point of view: name=36_nn
use_best_param: True, adjusted result: 1.5780000000000003
 number  value  params_fen_skip_train  params_fen_skip_val  params_lambda1  params_max_grad  params_random_move_count  params_random_move_minply  params_random_multi_pv  params_random_multi_pv_diff  params_skip_dupes_in_training  params_write_minply    state
      0  0.500                      0                    1            0.50              0.2                         5                          6                       4                          150                              1                    6 COMPLETE
      1  0.690                      1                    0            1.00              0.5                        11                          3                       4                          275                              1                    9 COMPLETE
      2  1.023                      0                    1            0.75              0.2                        16                          1                       4                          150                              1                   13 COMPLETE
      3  0.040                      1                    1            0.25              0.3                        11                          9                       4                          100                              0                   21 COMPLETE
      4  0.070                      0                    0            0.00              0.6                         8                          3                       7                          225                              0                    8 COMPLETE
      5  1.098                      1                    0            0.75              0.2                         5                          1                       4                          250                              0                   22 COMPLETE
      6  0.090                      0                    1            0.25              0.6                        14                          8                       5                          200                              1                   15 COMPLETE
      7  1.208                      0                    1            0.75              0.2                         9                          9                       6                          100                              0                   25 COMPLETE
      8  0.240                      1                    1            0.50              0.1                        17                          2                       4                          225                              0                   25 COMPLETE
      9  0.035                      1                    0            0.00              0.6                        20                          8                       8                          325                              1                   12 COMPLETE
     10  0.420                      0                    1            0.75              0.5                         8                         10                       6                          400                              0                   26 COMPLETE
     11  0.420                      1                    0            0.75              0.2                         5                          5                       6                          300                              0                   21 COMPLETE
     12  0.410                      1                    0            0.75              0.2                         8                          5                       7                          375                              0                   21 COMPLETE
     13  0.425                      0                    0            0.75              0.2                         5                          7                       5                          100                              0                   23 COMPLETE
     14  1.288                      1                    1            1.00              0.2                         7                         10                       5                          175                              0                   18 COMPLETE
     15  0.360                      0                    1            1.00              0.1                        10                         10                       5                          150                              0                   18 COMPLETE
     16  0.445                      0                    1            1.00              0.3                         8                         10                       7                          175                              0                   18 COMPLETE
     17  0.405                      1                    1            1.00              0.2                        13                          9                       6                          100                              0                   18 COMPLETE
     18  1.328                      1                    1            1.00              0.2                         7                          8                       5                          125                              0                   24 COMPLETE
     19  0.470                      1                    1            1.00              0.5                         6                          7                       5                          175                              0                   16 COMPLETE
     20  1.363                      1                    1            1.00              0.1                         7                          8                       5                          125                              0                   24 COMPLETE
     21  0.365                      1                    1            1.00              0.1                         7                          8                       5                          125                              0                   26 COMPLETE
     22  1.383                      1                    1            1.00              0.1                         7                          7                       5                          125                              0                   24 COMPLETE
     23  0.490                      1                    1            1.00              0.1                        10                          6                       5                          125                              0                   24 COMPLETE
     24  0.470                      1                    1            1.00              0.1                         6                          7                       5                          150                              0                   24 COMPLETE
     25  0.380                      1                    1            1.00              0.1                        10                          7                       6                          125                              0                   20 COMPLETE
     26  0.395                      1                    1            1.00              0.1                         7                          8                       5                          200                              0                   23 COMPLETE
     27  0.325                      1                    1            1.00              0.1                        12                          6                       5                          125                              0                   26 COMPLETE
     28  0.110                      1                    1            0.25              0.1                         9                          4                       6                          200                              0                   20 COMPLETE
     29  0.165                      1                    1            0.50              0.3                         6                          6                       4                          100                              1                   24 COMPLETE
     30  0.035                      1                    1            0.00              0.1                         5                          9                       4                          150                              0                   23 COMPLETE
     31  0.475                      1                    1            1.00              0.2                         7                          8                       5                          175                              0                   19 COMPLETE
     32  1.433                      1                    1            1.00              0.2                         7                          9                       5                          150                              0                   15 COMPLETE
     33  0.465                      1                    1            1.00              0.2                         9                          9                       5                          125                              0                   13 COMPLETE
     34  1.523                      1                    1            1.00              0.5                         6                          7                       4                          150                              1                   16 COMPLETE
     35  0.150                      1                    1            0.50              0.5                         6                          7                       4                          150                              1                   11 COMPLETE
     36  1.578                      1                    1            1.00              0.5                        11                          5                       4                          175                              1                   15 COMPLETE

```

##### Plots
Trials in x-axis and value in the y-axis. Value is from the result of engine vs engine match adjusted and sent to optimizer.
```
initial_best_value = 0.5
current_best_value = initial_best_value
trial 1 defeats trial 0 at 69% in a 100-game match.
current_best_value = current_best_value + (0.69 - initial_best_value) = 0.5 + (0.69-0.5) = 0.69
value of trial 1 = 0.69

trial 2 defeats trial 1 at 83.3%
current_best_value = current_best_value + (0.833 - initial_best_value) = 0.69 + (0.833-0.5) = 1.023
value of trial 2 is 1.023
```

We do this calculation everytime the new net defeats the current best at more than 50%. These are the values reported to the optimizer because the optimizer keeps track of the best value and other results.


![value](https://i.imgur.com/YQe8sTg.png)

***

max_grad is good at 0.5 at this point, high objective value is better, this is the result of engine vs engine match

![](https://i.imgur.com/XmiwSWY.png)

***

lambda at 1 is good

![](https://i.imgur.com/ufOyYCG.png)

***

random_multi_pv_diff is showing some promise in the range 100 to 200

![](https://i.imgur.com/7YLY8nP.png)

*** 

favored random_multi_pv values, a parameter during [training/validation positions generation](https://github.com/nodchip/Stockfish/blob/master/docs/gensfen.md)

```
random_multi_pv - the number of PVs used for determining the random move.
  If not specified then a truly random move will be chosen. If specified then 
  a multiPV search will be performed the random move will be one of the moves chosen by the search.
```

![](https://i.imgur.com/mhaWDXJ.png)

***

smart_fen_skipping has to be enabled, a parameter during learning

```
smart_fen_skipping - this is a flag option. 
  When specified some position that are not good candidates for teaching are skipped.
  This includes positions where the best move is a capture or promotion, and position
  where a king is in check. Default: 0.
```

![](https://i.imgur.com/etiyCy7.png)


## Credits
* [Nodchip](https://github.com/nodchip/Stockfish)
* [Stockfish](https://github.com/official-stockfish/Stockfish)
* [Optuna](https://github.com/optuna/optuna)
* [Cutechess](https://github.com/cutechess/cutechess)
