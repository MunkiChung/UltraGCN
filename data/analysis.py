import numpy as np

for bmk in ["CRSP/2006", "CRSP/2015", "THOMSON13f/2006", "THOMSON13f/2015", "amazon", "amazoncds", "electronics",
            "gowalla", "ml-1m", "yelp2018"]:
    count = 0

    n = 0
    with open(f"{bmk}/train.txt", "r") as f:
        for line in f:
            line = line.split(" ")
            user = int(line[0])
            items = np.array(line[1:]).astype(int)
            if max(items) >= n:
                n = max(items)
            count += len(items) - 1

    m = user + 1

    print(f"{bmk} - num users: {m}, num items: {n}, sparsity: {(count / m / n):.03f}")
