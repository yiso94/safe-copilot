TRAIN_TEST_SPLIT=navtest
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="./data/maps"
export NAVSIM_EXP_ROOT="./logs"
export NAVSIM_DEVKIT_ROOT="."
export OPENSCENE_DATA_ROOT="./data"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

uv run python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH