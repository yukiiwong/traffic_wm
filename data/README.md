# Data Directory

This directory contains trajectory data at different processing stages.

## Structure

```
data/
├── raw/              # Original CSV files from drones
├── processed/        # Preprocessed episode tensors
└── sample/           # Small test subset for quick experiments
```

## Raw Data (`raw/`)

Place your drone trajectory CSV files here. Each CSV should contain:

### Required Columns
- `track_id`: Unique vehicle identifier (integer)
- `frame`: Frame number / time index (integer)
- `center_x`: X position in meters (float)
- `center_y`: Y position in meters (float)
- `angle`: Heading angle in radians (float)
- `class_id`: Vehicle type identifier (float/int)

### Optional Columns (can be computed)
- `vx`: X velocity in m/s (float)
- `vy`: Y velocity in m/s (float)

### Example Format
```csv
track_id,frame,center_x,center_y,vx,vy,angle,class_id
1,0,10.5,20.3,2.1,0.5,1.57,2.0
1,1,12.6,20.8,2.1,0.5,1.57,2.0
2,0,15.2,18.7,1.8,-0.3,0.78,2.0
...
```

## Processed Data (`processed/`)

After running preprocessing, this directory contains:

- `episodes.npz`: All episodes in tensor format
  - `states`: [N, T, K, F] vehicle states
  - `masks`: [N, T, K] validity masks
  - `scene_ids`: [N] scene identifiers

- `normalization_stats.npz`: Normalization statistics
  - `mean`: [F] feature means
  - `std`: [F] feature standard deviations

## Sample Data (`sample/`)

A small subset of data for:
- Quick testing during development
- Verifying pipeline functionality
- Tutorial examples

## Preprocessing

To convert raw CSV to processed episodes:

```bash
python ../src/data/preprocess.py \
    --input_dir ./raw \
    --output_dir ./processed \
    --episode_length 30 \
    --max_vehicles 50 \
    --overlap 5
```

## Data Statistics

After preprocessing, check data statistics:

```python
import numpy as np

data = np.load('processed/episodes.npz')
states = data['states']  # [N, T, K, F]

print(f"Number of episodes: {states.shape[0]}")
print(f"Episode length (T): {states.shape[1]}")
print(f"Max vehicles (K): {states.shape[2]}")
print(f"Features (F): {states.shape[3]}")
```

## Notes

- Raw CSV files are not tracked in git (see `.gitignore`)
- Processed files are also excluded to save space
- Share preprocessed data separately if needed for reproducibility
