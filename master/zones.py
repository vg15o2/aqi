"""
Geometry helpers — point-in-polygon and tripwire crossing.
"""


def in_poly(pt, poly) -> bool:
    """Ray-casting point-in-polygon test."""
    if not poly or len(poly) < 3:
        return False
    x, y = pt
    inside = False
    px, py = poly[-1]
    for nx, ny in poly:
        if ((ny > y) != (py > y)) and x < (px - nx) * (y - ny) / (py - ny + 1e-10) + nx:
            inside = not inside
        px, py = nx, ny
    return inside


def cross_sign(px, py, ax, ay, bx, by) -> float:
    """Signed cross product of (AB) x (AP)."""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def side(px, py, ax, ay, bx, by) -> int:
    """Which side of line AB is point P?  +1 / -1 / 0."""
    c = cross_sign(px, py, ax, ay, bx, by)
    return 1 if c > 0 else (-1 if c < 0 else 0)
