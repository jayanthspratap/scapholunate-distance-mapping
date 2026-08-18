import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib
from scipy.ndimage import center_of_mass
from pydicom import dcmread

def dicom_to_gray(ds):
    raw = ds.pixel_array.astype(float)
    raw = (np.maximum(raw, 0) / raw.max()) * 255.0
    return np.uint8(raw)

def process_joint_space(nifti_path, pixel_spacing, slice_thickness, patient_id, outdir='output', 
                        threshold_mm_margin=1, threshold_mm=7.0, flip=False, save_fig=False):
    os.makedirs(outdir, exist_ok=True)

    img = nib.load(nifti_path).get_fdata()
    img = np.flip(np.transpose(img, (1, 0, 2)), axis=0)
    if flip:
        img = np.flip(img, axis=1)

    # Step 1: Determine central slice based on max scaphoid volume
    scaphoid_volumes = []
    scaphoid_rightmost_pts = []

    for i in range(img.shape[2]):
        segmented = img[:, :, i].astype(np.uint8) * 255
        _, thresh = cv2.threshold(segmented, 100, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        components = sorted(
            [j for j in range(1, num_labels) if stats[j, cv2.CC_STAT_AREA] > 100],
            key=lambda j: stats[j, cv2.CC_STAT_LEFT]
        )
        if len(components) >= 1:
            mask = (labels == components[0]).astype(np.uint8)
            scaphoid_volumes.append(np.sum(mask))

            contour, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contour:
                contour_pts = np.squeeze(contour[0])
                if contour_pts.ndim == 2:
                    rightmost_idx = np.argmax(contour_pts[:, 0])
                    rightmost_pt = contour_pts[rightmost_idx]
                    scaphoid_rightmost_pts.append((rightmost_pt[0], rightmost_pt[1]))
                else:
                    scaphoid_rightmost_pts.append((np.nan, np.nan))
            else:
                scaphoid_rightmost_pts.append((np.nan, np.nan))
        else:
            scaphoid_volumes.append(0)
            scaphoid_rightmost_pts.append((np.nan, np.nan))

    z_center_idx = int(np.argmax(scaphoid_volumes))
    x_center, y_center = scaphoid_rightmost_pts[z_center_idx]

    # Step 2: Process each slice
    segmented_coronals = []
    distances = []
    surfaces = []
    surfaces_transformed = []

    for i in range(img.shape[2]):
        segmented = img[:, :, i].astype(np.uint8) * 255
        _, thresh = cv2.threshold(segmented, 100, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        segmented_coronals.append(segmented)
        distances.append([])
        surfaces.append(([], []))
        surfaces_transformed.append(([], []))

        components = sorted(
            [j for j in range(1, num_labels) if stats[j, cv2.CC_STAT_AREA] > 100],
            key=lambda j: stats[j, cv2.CC_STAT_LEFT]
        )
        if len(components) < 2:
            continue

        scaphoid_mask = (labels == components[0]).astype(np.uint8)
        lunate_mask = (labels == components[1]).astype(np.uint8)

        scaphoid_contours, _ = cv2.findContours(scaphoid_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lunate_contours, _ = cv2.findContours(lunate_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not scaphoid_contours or not lunate_contours:
            continue

        scaphoid_contour = max(scaphoid_contours, key=cv2.contourArea)
        lunate_contour = max(lunate_contours, key=cv2.contourArea)

        scaphoid_com = center_of_mass(scaphoid_mask)
        lunate_com = center_of_mass(lunate_mask)

        scaphoid_points = np.squeeze(scaphoid_contour)
        lunate_points = np.squeeze(lunate_contour)

        if scaphoid_points.ndim != 2 or lunate_points.ndim != 2:
            continue

        scaphoid_surface = scaphoid_points[scaphoid_points[:, 0] > scaphoid_com[1]]
        lunate_surface = lunate_points[lunate_points[:, 0] < lunate_com[1]]

        if scaphoid_surface.size == 0 or lunate_surface.size == 0:
            continue

        y_range = (
            max(scaphoid_surface[:, 1].min(), lunate_surface[:, 1].min()),
            min(scaphoid_surface[:, 1].max(), lunate_surface[:, 1].max())
        )
        if y_range[1] <= y_range[0]:
            continue

        sca_pts = []
        lun_pts = []
        dists = []

        for y in range(int(y_range[0]), int(y_range[1])):
            sca_row = scaphoid_mask[y, :]
            lun_row = lunate_mask[y, :]
            if not sca_row.any() or not lun_row.any():
                continue
            sca_x = len(sca_row) - np.argmax(sca_row[::-1] > 0) - 1
            lun_x = np.argmax(lun_row > 0)
            d_mm = (lun_x - sca_x) * pixel_spacing
            sca_pts.append([sca_x, y])
            lun_pts.append([lun_x, y])
            dists.append(d_mm)

        if len(dists) == 0:
            continue

        sca_pts = np.array(sca_pts)
        lun_pts = np.array(lun_pts)
        dists = np.array(dists)
        mask = dists <= min(min(dists) + threshold_mm_margin, threshold_mm)

        if not np.any(mask):
            continue

        sca_pts = sca_pts[mask]
        lun_pts = lun_pts[mask]
        dists = dists[mask]

        if sca_pts.size == 0 or lun_pts.size == 0:
            continue

        distances[i] = dists.tolist()
        surfaces[i] = (sca_pts, lun_pts)

        z_mm = (i - z_center_idx) * slice_thickness
        sca_trans = [
            [(pt[0] - x_center) * pixel_spacing, (pt[1] - y_center) * pixel_spacing, z_mm]
            for pt in sca_pts
        ]
        lun_trans = [
            [(pt[0] - x_center) * pixel_spacing, (pt[1] - y_center) * pixel_spacing, z_mm]
            for pt in lun_pts
        ]
        surfaces_transformed[i] = (sca_trans, lun_trans)

    scaphoid_3d = [pt for slice_pts in surfaces_transformed for pt in slice_pts[0]]
    lunate_3d = [pt for slice_pts in surfaces_transformed for pt in slice_pts[1]]

    pd.DataFrame(scaphoid_3d).to_csv(f"{outdir}/{patient_id}_scaphoid_surface.csv", index=False, header=False)
    pd.DataFrame(lunate_3d).to_csv(f"{outdir}/{patient_id}_lunate_surface.csv", index=False, header=False)

    # Recalculate distances in 3D space and save to CSV
    distance_rows = []
    for z_idx, (sca_pts, lun_pts) in enumerate(surfaces_transformed):
        if not sca_pts or not lun_pts or len(sca_pts) != len(lun_pts):
            continue
        for i in range(len(sca_pts)):
            dist_x = np.linalg.norm(np.array(lun_pts[i]) - np.array(sca_pts[i]))
            y_centered = sca_pts[i][1]  # y in centered coordinates
            z_centered = sca_pts[i][2]  # z in centered coordinates
            distance_rows.append([dist_x, y_centered, z_centered])

    pd.DataFrame(distance_rows).to_csv(f"{outdir}/{patient_id}_distances.csv", index=False, header=False)

    if save_fig:
        slices_to_display = [i for i, (sca, lun) in enumerate(surfaces) if len(sca) > 0 or len(lun) > 0]
        num_slices = len(slices_to_display)

        distances_flat = [d for i in slices_to_display for d in distances[i] if len(distances[i]) > 0]
        mean_distance = np.mean(distances_flat) if distances_flat else 0

        grid_width = 3
        grid_height = (num_slices + grid_width - 1) // grid_width

        fig, axes = plt.subplots(grid_height, grid_width, figsize=(15, 5 * grid_height))
        fig.suptitle(f"({patient_id}) Mean Distance: {mean_distance:.2f} mm", fontsize=16, y=0.92)

        for ax in axes.flat:
            ax.axis('off')

        for plot_idx, slice_idx in enumerate(slices_to_display):
            ax = axes.flat[plot_idx]
            img = segmented_coronals[slice_idx]
            sca, lun = surfaces[slice_idx]

            all_points = np.vstack([sca, lun]) if len(sca) > 0 and len(lun) > 0 else (
                sca if len(sca) > 0 else lun
            )
            margin = 100
            x_min = max(int(all_points[:, 0].min()) - margin, 0)
            x_max = min(int(all_points[:, 0].max()) + margin, img.shape[1])
            y_min = max(int(all_points[:, 1].min()) - margin, 0)
            y_max = min(int(all_points[:, 1].max()) + margin, img.shape[0])

            cropped = img[y_min:y_max, x_min:x_max]
            ax.imshow(cropped, cmap='gray')

            if len(sca) > 0:
                sca = np.array(sca)
                ax.plot(sca[:, 0] - x_min, sca[:, 1] - y_min, 'bo', markersize=4)
            if len(lun) > 0:
                lun = np.array(lun)
                ax.plot(lun[:, 0] - x_min, lun[:, 1] - y_min, 'ro', markersize=4)

            ax.set_title(f"Slice {slice_idx}")

        plt.savefig(f"{outdir}/{patient_id}_slices.png", bbox_inches='tight')
        plt.close()

    return distances, segmented_coronals, surfaces_transformed
