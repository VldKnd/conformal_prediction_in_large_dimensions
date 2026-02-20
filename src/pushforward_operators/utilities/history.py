def safe_mean(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    
    return float(sum(values) / len(values))


def rolling_mean(values: list[float], window: int = 10, default: float = 0.0) -> float:
    if window <= 0:
        raise ValueError("window must be positive")
    elif window == -1:
        return safe_mean(values, default=default)

    return safe_mean(values[-window:], default=default)
