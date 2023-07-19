import numpy as np
import pandas as pd
import random

from dataval.dataset import WeatherDataset

@staticmethod
def corrupt_null_by_feature(df, sensor_group, corruption_rate=0.1, random_state=42):
    sensor_cols = [col for col in df.columns if sensor_group in col]
    copy = df.copy()
    copy[sensor_cols] = np.where(
        np.random.default_rng(seed=random_state).uniform(size=copy[sensor_cols].shape) < corruption_rate,
        float("nan"),
        copy[sensor_cols]
    )
    return copy, sensor_cols

@staticmethod
def corrupt_nonnegative_by_feature(df, sensor_group, corruption_rate=0.1, random_state=42):
    summary = df.describe()
    sensor_cols = [
        col
        for col in df.columns
        if sensor_group in col and summary[col].min() >= 0
    ]
    copy = df.copy()

    # Set to random negative value
    copy[sensor_cols] = np.where(
        np.random.default_rng(seed=random_state).uniform(size=copy[sensor_cols].shape) < corruption_rate,
        -1 * random.randrange(0, 100),
        copy[sensor_cols]
    )
    return copy, sensor_cols

@staticmethod
def corrupt_typecheck_by_feature(df, sensor_group, corruption_rate=0.1, random_state=42):
    sensor_cols = [
        col
        for col in df.columns
        if sensor_group in col and all(x.is_integer() for x in df[col])
    ]
    copy = df.copy()

    # Set to random float
    copy[sensor_cols] = np.where(
        np.random.default_rng(seed=random_state).uniform(size=copy[sensor_cols].shape) < corruption_rate,
        random.random() * 100,
        copy[sensor_cols]
    )
    return copy, sensor_cols

@staticmethod
def corrupt_units_by_feature(df, sensor_group, corruption_rate=0.1, random_state=42):
    sensor_cols = [col for col in df.columns if sensor_group in col]
    copy = df.copy()

    # Change from Celsius to Fahrenheit
    copy[sensor_cols] = np.where(
        np.random.default_rng(seed=random_state).uniform(size=copy[sensor_cols].shape) < corruption_rate,
        (9 / 5) * copy[sensor_cols] + 32,
        copy[sensor_cols]
    )
    return copy, sensor_cols

@staticmethod
def corrupt_average_by_feature(df, sensor_group, corruption_rate=0.1, random_state=42):
    sensor_cols = [col for col in df.columns if sensor_group in col]
    copy = df.copy()

    # Set to average of the sensors
    copy[sensor_cols] = np.where(
        np.random.default_rng(seed=random_state).uniform(size=copy[sensor_cols].shape) < corruption_rate,
        np.asarray(np.mean(copy[sensor_cols], axis=1))[:, np.newaxis],
        copy[sensor_cols]
    )
    return copy, sensor_cols

@staticmethod
def corrupt_pinned_by_feature(df, sensor_group, corruption_rate=0.1, pinned_value=5, random_state=42):
    sensor_cols = [col for col in df.columns if sensor_group in col]
    copy = df.copy()

    # Set to pinned value
    copy[sensor_cols] = np.where(
        np.random.default_rng(seed=random_state).uniform(size=copy[sensor_cols].shape) < corruption_rate,
        pinned_value,
        copy[sensor_cols]
    )
    return copy, sensor_cols

@staticmethod
def iterate_corruptions_by_feature(df, sensor_group, **kwargs):
    for corruption in [
        WeatherDataset.corrupt_null,
        WeatherDataset.corrupt_nonnegative,
        WeatherDataset.corrupt_typecheck,
        WeatherDataset.corrupt_units,
        WeatherDataset.corrupt_average,
        WeatherDataset.corrupt_pinned,
        WeatherDataset.corrupt_null_by_feature,
        WeatherDataset.corrupt_nonnegative_by_feature,
        WeatherDataset.corrupt_typecheck_by_feature,
        WeatherDataset.corrupt_units_by_feature,
        WeatherDataset.corrupt_average_by_feature,
        WeatherDataset.corrupt_pinned_by_feature,
    ]:
        yield corruption.__name__, corruption(df, sensor_group, **kwargs)

WeatherDataset.corrupt_null_by_feature = corrupt_null_by_feature
WeatherDataset.corrupt_nonnegative_by_feature = corrupt_nonnegative_by_feature
WeatherDataset.corrupt_typecheck_by_feature = corrupt_typecheck_by_feature
WeatherDataset.corrupt_units_by_feature = corrupt_units_by_feature
WeatherDataset.corrupt_average_by_feature = corrupt_average_by_feature
WeatherDataset.corrupt_pinned_by_feature = corrupt_pinned_by_feature
WeatherDataset.iterate_corruptions_by_feature = iterate_corruptions_by_feature