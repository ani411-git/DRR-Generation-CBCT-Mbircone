import numpy as np
from scipy.optimize import minimize
import cv2
from ssnhatesme import generate_drr_mbircone, get_projection_matrix, decompose_projection_matrix, load_ct, parse_projection_matrix
from scipy.spatial.transform import Rotation

def ncc(img1, img2):
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1 -= np.mean(img1)
    img2 -= np.mean(img2)
    img1 /= (np.std(img1) + 1e-8)
    img2 /= (np.std(img2) + 1e-8)
    return np.sum(img1 * img2) / img1.size

def get_camera_center(P):
    Q = P[:, :3]
    m = P[:, 3]
    C = -np.linalg.inv(Q) @ m
    return C

def get_principal_ray(P, K, width, height):
    # Detector center in pixel coordinates
    detector_center_px = np.array([width/2, height/2, 1.0])
    # Direction in camera coordinates
    direction_cam = np.linalg.inv(K) @ detector_center_px
    direction_cam = direction_cam / np.linalg.norm(direction_cam)
    # Get rotation matrix from P
    _, R, _, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    direction_world = R.T @ direction_cam
    direction_world = direction_world / np.linalg.norm(direction_world)
    return direction_world

def closest_point_between_lines(p1, d1, p2, d2):
    # p1, p2: points on each line
    # d1, d2: direction vectors (should be normalized)
    cross = np.cross(d1, d2)
    denom = np.linalg.norm(cross)**2
    if denom < 1e-8:
        # Lines are parallel
        return (p1 + p2) / 2
    t = np.dot(np.cross((p2 - p1), d2), cross) / denom
    s = np.dot(np.cross((p2 - p1), d1), cross) / denom
    closest1 = p1 + d1 * t
    closest2 = p2 + d2 * s
    return (closest1 + closest2) / 2

def registration_objective(params, fixed_xray_ap, fixed_xray_lp, volume_attenuation, volume_spacing,
                          det_size_x, det_size_y, det_spacing_x, det_spacing_y,
                          K_ap, K_lp, SAD_ap, SID_ap, SAD_lp, SID_lp):
    rx, ry, rz, tx, ty, tz = params
    euler_angles_deg = [rx, ry, rz]
    translation_vec = [tx, ty, tz]
    # AP view
    P_ap = get_projection_matrix(euler_angles_deg, translation_vec, K_ap)
    _, SDD_ap_new, _, _, _, _, sourcetiso_ap_new = decompose_projection_matrix(P_ap)
    SAD_ap_new = sourcetiso_ap_new
    SID_ap_new = SDD_ap_new
    drr_ap = generate_drr_mbircone(volume_attenuation, volume_spacing, SAD_ap_new, SID_ap_new, det_size_x, det_size_y, det_spacing_x, det_spacing_y, rotation_angle_deg=rz)
    drr_ap = (drr_ap - np.min(drr_ap)) / (np.max(drr_ap) - np.min(drr_ap) + 1e-8)
    fixed_ap = (fixed_xray_ap - np.min(fixed_xray_ap)) / (np.max(fixed_xray_ap) - np.min(fixed_xray_ap) + 1e-8)
    ncc_ap = ncc(fixed_ap, drr_ap)
    # LP view
    P_lp = get_projection_matrix(euler_angles_deg, translation_vec, K_lp)
    _, SDD_lp_new, _, _, _, _, sourcetiso_lp_new = decompose_projection_matrix(P_lp)
    SAD_lp_new = sourcetiso_lp_new
    SID_lp_new = SDD_lp_new
    drr_lp = generate_drr_mbircone(volume_attenuation, volume_spacing, SAD_lp_new, SID_lp_new, det_size_x, det_size_y, det_spacing_x, det_spacing_y, rotation_angle_deg=rz)
    # Flip LP DRR horizontally for neurological convention (source on patient's left)
    drr_lp = np.fliplr(drr_lp)
    drr_lp = (drr_lp - np.min(drr_lp)) / (np.max(drr_lp) - np.min(drr_lp) + 1e-8)
    fixed_lp = (fixed_xray_lp - np.min(fixed_xray_lp)) / (np.max(fixed_xray_lp) - np.min(fixed_xray_lp) + 1e-8)
    ncc_lp = ncc(fixed_lp, drr_lp)
    return -(ncc_ap + ncc_lp) / 2.0

def main():
    # === USER: Set your paths here ===
    ct_folder = '/Users/aravindr/Downloads/A_CT_SCAN_FOLDER'  # Folder containing CT DICOM series
    proj_matrix_file = '/Users/aravindr/Downloads/P_MATRIX_FILE.txt'  # File with both AP and LP projection matrices
    ap_xray_path = '/Users/aravindr/Documents/HTIC/mbircone/output/APFlip.png'  # AP X-ray image
    lp_xray_path = '/Users/aravindr/Documents/HTIC/mbircone/output/LPFlip.png'  # LP X-ray image

    # --- Load CT volume and spacing ---
    volume, volume_spacing, origin, metadata = load_ct(ct_folder)
    
    # --- Convert CT from LPS to RAS and to attenuation coefficients (same as ssnhatesme.py) ---
    # Convert from LPS to RAS by flipping the first two spatial axes (Y and X)
    volume_ras = volume[:, ::-1, ::-1]
    # Convert to attenuation coefficients
    mu_water_at_80kev = 0.02
    volume_attenuation = ((volume_ras + 1000.0) / 1000.0) * mu_water_at_80kev
    volume_attenuation[volume_attenuation < 0] = 0
    print("CT volume converted from LPS to RAS and to attenuation coefficients")

    # --- Load AP and LP X-ray images ---
    fixed_xray_ap = cv2.imread(ap_xray_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    fixed_xray_lp = cv2.imread(lp_xray_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # --- Parse projection matrices ---
    # Use hardcoded P1 and P2 matrices from ssnhatesme.py
    P1 = np.array([
        [-4298.936264, 801.573106, 504.814267, 758951.144144],
        [-957.239585, -2682.742358, -3351.795366, -201860.747034],
        [-0.207768, -0.806154, 0.554029, 815.235148]
    ], dtype=np.float64)
    P2 = np.array([
        [441.718242, 3523.733037, -2033.378096, 217409.124851],
        [-1160.510253, -1774.851122, -3484.694510, -199449.148331],
        [-0.948070, 0.317982, 0.007080, 745.199857]
    ], dtype=np.float64)
    # By your convention, P1 = LP, P2 = AP
    P_lp = P1
    P_ap = P2

    # --- Extract K matrices and compute SAD/SID from projection matrices ---
    source_ap, SDD_ap, detector_center_ap, R_ap, translation_norm_ap, K_ap, sourcetiso_ap = decompose_projection_matrix(P_ap)
    source_lp, SDD_lp, detector_center_lp, R_lp, translation_norm_lp, K_lp, sourcetiso_lp = decompose_projection_matrix(P_lp)
    
    # SDD (Source-Detector Distance) is equivalent to SID (Source-Image Distance)
    # sourcetiso is equivalent to SAD (Source-Axis Distance)
    SID_ap = SDD_ap
    SID_lp = SDD_lp
    SAD_ap = sourcetiso_ap
    SAD_lp = sourcetiso_lp
    
    print(f"AP view - SAD: {SAD_ap:.2f} mm, SID: {SID_ap:.2f} mm")
    print(f"LP view - SAD: {SAD_lp:.2f} mm, SID: {SID_lp:.2f} mm")

    # --- Compute C-arm isocenter from projection matrices ---
    detector_width = 1052
    detector_height = 1024
    
    # Get source positions
    source_ap = get_camera_center(P_ap)
    source_lp = get_camera_center(P_lp)
    
    # Get principal ray directions
    dir_ap = get_principal_ray(P_ap, K_ap, detector_width, detector_height)
    dir_lp = get_principal_ray(P_lp, K_lp, detector_width, detector_height)
    
    # Compute isocenter
    c_arm_isocenter = closest_point_between_lines(source_ap, dir_ap, source_lp, dir_lp)
    print("Computed C-arm isocenter:", c_arm_isocenter)

    # --- Set detector parameters (update if needed) ---
    det_size_x = 1052
    det_size_y = 1024
    det_spacing_x = 0.741
    det_spacing_y = 0.741
    x0 = [32.3818908, -7.81284325, -169.96967506, 194.63683, 157.27507, 299.9288]  # Initial guess from ssnhatesme.py

    res = minimize(
        registration_objective, x0,
        args=(fixed_xray_ap, fixed_xray_lp, volume_attenuation, volume_spacing, det_size_x, det_size_y, det_spacing_x, det_spacing_y, K_ap, K_lp, SAD_ap, SID_ap, SAD_lp, SID_lp),
        method='Powell', options={'maxiter': 5, 'disp': True}
    )
    print('Best parameters:', res.x)
    print('Best NCC:', -res.fun)
    print('Copy these rx, ry, rz, tx, ty, tz values into ssnhatesme.py for best alignment.')
    print('C-arm isocenter for reference:', c_arm_isocenter)

if __name__ == "__main__":
    main()
