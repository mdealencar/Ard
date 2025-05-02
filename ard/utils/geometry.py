import numpy as np
import jax.numpy as jnp

def ray_casting_boundary(
    boundary_vertices,
    boundary_normals,
    turbine_x,
    turbine_y,
    discrete=False,
    s=700,
    tol=1e-6,
    return_region=False,
    regions=None,
    c=None,
):
    """
    Calculate the distance from each turbine to the nearest point on the boundary using
    the ray-casting algorithm. Negative means the turbine is inside the boundary.
    This implementation is based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl)

    Args:
        boundary_vertices (np.ndarray or list of np.ndarray): Vertices of the boundary in
            counterclockwise order. If `discrete` is True, this should be a list of arrays.
        boundary_normals (np.ndarray or list of np.ndarray): Unit normal vectors for each
            boundary face in counterclockwise order. If `discrete` is True, this should be
            a list of arrays.
        turbine_x (np.ndarray): Turbine x locations.
        turbine_y (np.ndarray): Turbine y locations.
        discrete (bool, optional): If True, indicates the boundary is made of multiple discrete regions.
            Defaults to False.
        s (float, optional): Smoothing factor for smooth max. Defaults to 700.
        tol (float, optional): Tolerance for determining proximity. Defaults to 1e-6.
        return_region (bool, optional): If True, return a vector specifying which region each turbine is in.
            Defaults to False.
        regions (list, optional): Predefined region assignments for turbines. Defaults to None.
        c (np.ndarray, optional): Preallocated array for constraint values. Defaults to None.

    Returns:
        np.ndarray: Constraint values for each turbine.
        np.ndarray (optional): Region assignments for each turbine (if `return_region` is True).
    """
    # Number of turbines
    nturbines = len(turbine_x)

    # Initialize constraint output values
    if c is None:
        c = np.zeros(nturbines, dtype=float)

    # Single region
    if not discrete:
        for i in range(nturbines):
            # Determine if the point is contained in the polygon
            c[i] = point_in_polygon(
                np.array([turbine_x[i], turbine_y[i]]),
                boundary_vertices,
                boundary_normals,
                s=s,
                shift=tol,
            )
        return c

    # Multiple discrete regions with predefined region assignments
    elif regions is not None and len(regions) > 0:
        for i in range(nturbines):
            # Determine if the point is contained in the assigned polygonal region
            c[i] = point_in_polygon(
                np.array([turbine_x[i], turbine_y[i]]),
                boundary_vertices[regions[i]],
                boundary_normals[regions[i]],
                s=s,
                shift=tol,
            )
        return c

    # Multiple discrete regions without predefined region assignments
    else:
        # Number of regions
        nregions = len(boundary_vertices)

        # Initialize region and status arrays
        region = np.zeros(nturbines, dtype=int)
        status = np.zeros(nturbines, dtype=int)

        # Initialize array to hold distances from each turbine to the closest boundary face
        turbine_to_region_distance = np.zeros(nregions, dtype=float)

        for i in range(nturbines):
            # Iterate through each region
            for k in range(nregions):
                # Check if the point is in this region
                ctmp = point_in_polygon(
                    np.array([turbine_x[i], turbine_y[i]]),
                    boundary_vertices[k],
                    boundary_normals[k],
                    s=s,
                    shift=tol,
                    return_distance=False,
                )
                if ctmp <= 0:  # Negative if in boundary
                    c[i] = point_in_polygon(
                        np.array([turbine_x[i], turbine_y[i]]),
                        boundary_vertices[k],
                        boundary_normals[k],
                        s=s,
                        shift=tol,
                        return_distance=True,
                    )
                    status[i] = 1
                    if return_region:
                        region[i] = k
                    break

            # Check if the turbine is in none of the regions
            if status[i] == 0:
                for k in range(nregions):
                    # Calculate distance to each region
                    turbine_to_region_distance[k] = point_in_polygon(
                        np.array([turbine_x[i], turbine_y[i]]),
                        boundary_vertices[k],
                        boundary_normals[k],
                        s=s,
                        shift=tol,
                        return_distance=True,
                    )

                # Magnitude of the constraint value
                c[i] = -smooth_max(-turbine_to_region_distance, s=s)

                # Set status to indicate that the turbine has been assigned
                status[i] = 1

                # Indicate closest region
                region[i] = np.argmin(turbine_to_region_distance)

        if return_region:
            return c, region
        return c

def boundary_normals_calculator(boundary_vertices, nboundaries=1):
    """
    Calculate unit vectors perpendicular to each edge of each polygon in a set of polygons.
    This implementation based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl).

    Args:
        boundary_vertices (list of np.ndarray): List of m-by-2 arrays, where each array contains
            the boundary vertices of a polygon in counterclockwise order.
        nboundaries (int, optional): The number of boundaries in the set. Defaults to 1.

    Returns:
        list of np.ndarray: List of m-by-2 arrays of unit vectors perpendicular to each edge
            of each polygon.
    """
    if nboundaries == 1:
        return single_boundary_normals_calculator(boundary_vertices)
    else:
        boundary_normals = []
        for i in range(nboundaries):
            normals = single_boundary_normals_calculator(boundary_vertices[i])
            boundary_normals.append(normals)
        return boundary_normals
 
def single_boundary_normals_calculator(boundary_vertices):
    """
    Calculate unit vectors perpendicular to each edge of a polygon.This implementation 
    is based on FLOWFarm.jl (https://github.com/byuflowlab/FLOWFarm.jl).

    Args:
        boundary_vertices (np.ndarray): m-by-2 array containing all the boundary vertices
            in counterclockwise order.

    Returns:
        np.ndarray: m-by-2 array of unit vectors perpendicular to each edge of the polygon.
    """
    
    # Get the number of vertices in the polygon
    nvertices = boundary_vertices.shape[0]

    # Add the first vertex to the end to form a closed loop
    boundary_vertices = np.vstack([boundary_vertices, boundary_vertices[0]])

    # Initialize an array to hold boundary normals
    boundary_normals = np.zeros((nvertices, 2))

    # Iterate over each boundary edge
    for i in range(nvertices):
        # Create a vector normal to the boundary
        dx = boundary_vertices[i + 1, 0] - boundary_vertices[i, 0]
        dy = boundary_vertices[i + 1, 1] - boundary_vertices[i, 1]
        boundary_normals[i, :] = [-dy, dx]

        # Normalize the vector
        norm = np.linalg.norm(boundary_normals[i, :])
        if norm > 0:
            boundary_normals[i, :] /= norm

    return boundary_normals

def point_on_line(p: np.ndarray, v1: np.ndarray, v2: np.ndarray, tol=1e-6):
    """
    Determine if a point lies on a line segment.

    Given a line determined by two points (v1 and v2), determine if the point (p) lies on the line
    between those points within a given tolerance.

    Args:
        p (np.ndarray): Point of interest (2D vector).
        v1 (np.ndarray): First vertex of the line (2D vector).
        v2 (np.ndarray): Second vertex of the line (2D vector).
        tol (float): Tolerance for determining co-linearity.

    Returns:
        bool: True if the point lies on the line, False otherwise.
    """

    d = distance_point_to_lineseg_nd(p, v1, v2)

    return jnp.isclose(d, 0.0, atol=tol / 2.0)

def point_in_polygon(
    point: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray = None,
    s: float = 700,
    shift: float = 1e-10,
    return_distance: bool = True,
):
    """
    Determine the signed distance from a point to a polygon.

    Given a polygon defined by a set of vertices, determine the signed distance from the point
    to the polygon. Returns the negative (-) distance if the point is inside or on the polygon,
    and positive (+) otherwise. If `return_distance` is False, returns -1 if the point is inside
    or on the boundary, and 1 otherwise. This implementation based on FLOWFarm.jl 
    (https://github.com/byuflowlab/FLOWFarm.jl)

    Args:
        point (np.ndarray): Point of interest (2D vector).
        vertices (np.ndarray): Vertices of the polygon (Nx2 array).
        normals (np.ndarray, optional): Normals of the polygon edges. If not provided, they will
            be calculated.
        s (float, optional): Smoothing factor for the smoothmax function. Defaults to 700.
        shift (float, optional): Small shift to handle edge cases. Defaults to 1e-10.
        return_distance (bool, optional): Whether to return the signed distance or just
            inside/outside status. Defaults to True.

    Returns:
        float: Signed distance or inside/outside status.
    """

    if return_distance and isinstance(point[0], int):
        raise ValueError("Point coordinates must be floats, not integers.")

    nvertices = vertices.shape[0]
    intersection_counter = 0
    turbine_to_face_distance = np.zeros(nvertices)

    # Add the first vertex to the end to close the polygon loop
    vertices = np.vstack([vertices, vertices[0]])

    # Flags for point status
    onvertex = False
    onedge = False

    # Check if the point is on a vertex or edge
    for i in range(nvertices):
        if np.allclose(point, vertices[i], atol=shift / 2.0):
            onvertex = True
            break
        elif point_on_line(point, vertices[i], vertices[i + 1], tol=shift / 2.0):
            onedge = True
            break

    # Iterate through each boundary edge
    for j in range(nvertices):
        # Check if the x-coordinate of the point is between the x-coordinates of the edge
        if (
            (vertices[j, 0] <= point[0] < vertices[j + 1, 0])
            or (vertices[j, 0] >= point[0] > vertices[j + 1, 0])
        ):
            # Calculate the y-coordinate of the edge at the x-coordinate of the point
            y = (
                (vertices[j + 1, 1] - vertices[j, 1])
                / (vertices[j + 1, 0] - vertices[j, 0])
                * (point[0] - vertices[j, 0])
                + vertices[j, 1]
            )
            if point[1] < y:
                intersection_counter += 1

        if return_distance:
            # Calculate the vector from the point to the second vertex of the edge
            turbine_to_second_facepoint = vertices[j + 1] - point

            # Calculate the vector defining the edge
            boundary_vector = vertices[j + 1] - vertices[j]

            # Check if perpendicular distance is the shortest
            if (
                np.dot(boundary_vector, -turbine_to_second_facepoint) > 0
                and np.dot(boundary_vector, turbine_to_second_facepoint) > 0
            ):
                d = np.dot(turbine_to_second_facepoint, normals[j])
                turbine_to_face_distance[j] = abs(d + shift if onedge or onvertex else d)
            elif np.dot(boundary_vector, -turbine_to_second_facepoint) < 0:
                turbine_to_face_distance[j] = np.linalg.norm(turbine_to_second_facepoint)
            else:
                turbine_to_face_distance[j] = np.linalg.norm(turbine_to_second_facepoint)

    if return_distance:
        c = -smooth_max(-turbine_to_face_distance, s=s)
        if intersection_counter % 2 == 1 or onvertex or onedge:
            c = -c
    else:
        c = -1 if intersection_counter % 2 == 1 or onvertex or onedge else 1

    return c