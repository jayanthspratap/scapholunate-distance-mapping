# Scapholunate distance mapping

Code and derived data for "Automated Three-Dimensional Mapping of Scapholunate Distance on CT Compared to
Radiograph Measurements". For every row of every coronal slice, `distance_mapping.py` measures the horizontal
radial-ulnar gap between the scaphoid and lunate margins; the per-slice measurements are assembled into a 3D
distance field. This is a row-wise horizontal gap, not a perpendicular nearest-surface distance.

## Data

`data/` holds the derived point clouds for all 29 processed patients: 25 cases with confirmed SLIL injury and
4 non-injured controls, split by the `Control` column of `res/dcm_metadata.csv`. The CT segmentations they
were computed from cannot be released. Patient IDs are the 8-digit pseudo-anonymous study IDs used in `res/`.

Three files per patient, no headers, row-for-row paired in the same order:

    <id>_scaphoid_surface.csv   x_scaphoid, Y, Z
    <id>_lunate_surface.csv     x_lunate,   Y, Z
    <id>_distances.csv          gap_mm,     Y, Z    (gap_mm = x_lunate - x_scaphoid)

The two surface files are the joint-facing margin points only, one point per image row per coronal slice;
they are not full bone surfaces. Within a slice, candidate distances are kept only if within 1.0 mm of that
slice's minimum gap and no greater than 7.0 mm.

Coordinates are in mm, relative to the scaphoid's rightmost margin point on the slice of maximum scaphoid
cross-sectional area: X = radial/ulnar (the gap direction), Y = proximal (-) to distal (+), Z = dorsal (+) to
palmar (-). Right wrists are mirrored to a common laterality.

## Running

    pip install -r requirements.txt

`01_setup_distance_mapping.ipynb` builds the `res/` metadata and runs the mapping over the NIfTI
segmentations, writing per-patient outputs to `output_case/` and `output_control/`; `data/` is those outputs
for all 29 patients combined. Without the segmentations it cannot be re-run, so start at
`02_analysis_post_mapping.ipynb`, which reads `data/` and `res/` and reproduces the correlations and figures.

Interactive viewer: https://jayanthspratap.github.io/scapholunate-distance-mapping/
