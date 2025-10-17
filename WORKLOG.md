### 2025-10-16

1. Properly implemented static slice is faster than dynamic slice (using cooperative tensor to accumulate and then write back after stream k done).
2. Dynamic extents seems to be faster than static extents (???).
3. 128x64x64 (and 4 simdgroups) seems to be the ideal tile size regardless for Neural Accelerators (if stick with static slice).
4. Seems most hyper-parameters are related to K (i.e. overlapping (stream-k) not as effective for tensor ops?).
5. Fully unroll K seems to be important.
6. Split K solves the issue with only stream K.
