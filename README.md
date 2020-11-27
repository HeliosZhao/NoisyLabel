# NoisyLabel


## How to Run

### Train
```console

python main.py -l sl --eta 0.6 [--data-dir ./data] [--logs-dir ./logs]

```

### Test
```console

python main.py -l sl --eta 0.6 --evaluate --resume "your checkpoint path" [--data-dir ./data] [--logs-dir ./logs]

```

## Results on CIFAR10
Result of best Epoch
| noise rate  | 0.0 | 0.2  | 0.4 | 0.6 | 0.8 |
| ------------------- |:---:|:---:|:---:|:---:|:---:|
| Baseline    |91.73|84.05|79.29|70.91|44.24|
| Noise Robust Loss   |91.56|89.69|86.60|80.19|54.62|
| Small Select Loss   |91.56|89.96|87.29|82.50|51.91|
