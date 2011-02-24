

def clamp(x, m = 0.0, M = 1.0):
    if (x<m):
        return m
    elif (x>M):
        return M
    else:
        return x


__all__ = ["clamp"]
