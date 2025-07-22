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
    
    # Apply Gaussian blur to reduce noise
    img1 = cv2.GaussianBlur(img1, (3, 3), 0)
    img2 = cv2.GaussianBlur(img2, (3, 3), 0)
    
    # Normalize
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

    # Try multiple starting points
    starting_points = [
        x0,  # Original guess
        [x0[0] + 10, x0[1], x0[2], x0[3], x0[4], x0[5]],  # Perturbed rx
        [x0[0], x0[1] + 10, x0[2], x0[3], x0[4], x0[5]],  # Perturbed ry
        [x0[0], x0[1], x0[2] + 10, x0[3], x0[4], x0[5]],  # Perturbed rz
        [x0[0], x0[1], x0[2], x0[3] + 20, x0[4], x0[5]],  # Perturbed tx
        [x0[0], x0[1], x0[2], x0[3], x0[4] + 20, x0[5]],  # Perturbed ty
        [x0[0], x0[1], x0[2], x0[3], x0[4], x0[5] + 20],  # Perturbed tz
    ]

    # Try multiple optimization methods
    methods = ['Powell', 'L-BFGS-B', 'TNC']
    best_result = None
    best_ncc = -float('inf')
    
    for start_point_idx, start_point in enumerate(starting_points):
        print(f"\n--- Starting point {start_point_idx + 1}/{len(starting_points)}: {start_point} ---")
        
        for method in methods:
            print(f"\n--- Trying {method} method with start point {start_point_idx + 1} ---")
            try:
                if method == 'Powell':
                    res = minimize(
                        registration_objective, start_point,
                        args=(fixed_xray_ap, fixed_xray_lp, volume_attenuation, volume_spacing, det_size_x, det_size_y, det_spacing_x, det_spacing_y, K_ap, K_lp, SAD_ap, SID_ap, SAD_lp, SID_lp),
                        method=method, options={'maxiter': 20, 'disp': False}
                    )
                else:
                    # For gradient-based methods, add bounds
                    bounds = [(-180, 180), (-180, 180), (-180, 180), (-500, 500), (-500, 500), (-500, 500)]
                    res = minimize(
                        registration_objective, start_point,
                        args=(fixed_xray_ap, fixed_xray_lp, volume_attenuation, volume_spacing, det_size_x, det_size_y, det_spacing_x, det_spacing_y, K_ap, K_lp, SAD_ap, SID_ap, SAD_lp, SID_lp),
                        method=method, bounds=bounds, options={'maxiter': 20, 'disp': False}
                    )
                
                ncc_score = -res.fun
                print(f"{method} NCC: {ncc_score:.6f}")
                
                if ncc_score > best_ncc:
                    best_ncc = ncc_score
                    best_result = res
                    print(f"*** New best NCC: {best_ncc:.6f} ***")
                    
            except Exception as e:
                print(f"{method} failed: {e}")
    
    res = best_result
    print('Best parameters:', res.x)
    print('Best NCC:', -res.fun)
    print('Copy these rx, ry, rz, tx, ty, tz values into ssnhatesme.py for best alignment.')
    print('C-arm isocenter for reference:', c_arm_isocenter)
    
    # Generate final DRRs with optimized parameters
    print("\n--- Generating final DRRs with optimized parameters ---")
    rx_opt, ry_opt, rz_opt, tx_opt, ty_opt, tz_opt = res.x
    euler_angles_deg_opt = [rx_opt, ry_opt, rz_opt]
    translation_vec_opt = [tx_opt, ty_opt, tz_opt]
    
    # Generate optimized projection matrices
    P_ap_opt = get_projection_matrix(euler_angles_deg_opt, translation_vec_opt, K_ap)
    P_lp_opt = get_projection_matrix(euler_angles_deg_opt, translation_vec_opt, K_lp)
    
    # Decompose to get SAD/SID for DRR generation
    _, SDD_ap_opt, _, _, _, _, sourcetiso_ap_opt = decompose_projection_matrix(P_ap_opt)
    _, SDD_lp_opt, _, _, _, _, sourcetiso_lp_opt = decompose_projection_matrix(P_lp_opt)
    SAD_ap_opt = sourcetiso_ap_opt
    SID_ap_opt = SDD_ap_opt
    SAD_lp_opt = sourcetiso_lp_opt
    SID_lp_opt = SDD_lp_opt
    
    print(f"Optimized AP - SAD: {SAD_ap_opt:.2f} mm, SID: {SID_ap_opt:.2f} mm")
    print(f"Optimized LP - SAD: {SAD_lp_opt:.2f} mm, SID: {SID_lp_opt:.2f} mm")
    
    # Generate final DRRs
    drr_ap_final = generate_drr_mbircone(volume_attenuation, volume_spacing, SAD_ap_opt, SID_ap_opt, 
                                        det_size_x, det_size_y, det_spacing_x, det_spacing_y, rotation_angle_deg=rz_opt)
    drr_lp_final = generate_drr_mbircone(volume_attenuation, volume_spacing, SAD_lp_opt, SID_lp_opt, 
                                        det_size_x, det_size_y, det_spacing_x, det_spacing_y, rotation_angle_deg=rz_opt)
    
    # Flip LP DRR for neurological convention
    drr_lp_final = np.fliplr(drr_lp_final)
    
    # Normalize for display
    drr_ap_final_norm = (drr_ap_final - np.min(drr_ap_final)) / (np.max(drr_ap_final) - np.min(drr_ap_final) + 1e-8)
    drr_lp_final_norm = (drr_lp_final - np.min(drr_lp_final)) / (np.max(drr_lp_final) - np.min(drr_lp_final) + 1e-8)
    
    # Convert to uint8 for saving
    drr_ap_final_uint8 = (drr_ap_final_norm * 255).astype(np.uint8)
    drr_lp_final_uint8 = (drr_lp_final_norm * 255).astype(np.uint8)
    
    # Save final DRRs
    import os
    output_dir = "DRR_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cv2.imwrite(os.path.join(output_dir, "AP_DRR_optimized.png"), drr_ap_final_uint8)
    cv2.imwrite(os.path.join(output_dir, "LP_DRR_optimized.png"), drr_lp_final_uint8)
    
    print(f"Final optimized DRRs saved to {output_dir}/")
    print(f"AP DRR: {output_dir}/AP_DRR_optimized.png")
    print(f"LP DRR: {output_dir}/LP_DRR_optimized.png")
    
    # Also save the original X-rays for comparison
    cv2.imwrite(os.path.join(output_dir, "AP_Xray_original.png"), fixed_xray_ap.astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "LP_Xray_original.png"), fixed_xray_lp.astype(np.uint8))
    
    # Calculate final NCC scores
    fixed_ap_norm = (fixed_xray_ap - np.min(fixed_xray_ap)) / (np.max(fixed_xray_ap) - np.min(fixed_xray_ap) + 1e-8)
    fixed_lp_norm = (fixed_xray_lp - np.min(fixed_xray_lp)) / (np.max(fixed_xray_lp) - np.min(fixed_xray_lp) + 1e-8)
    
    ncc_ap_final = ncc(fixed_ap_norm, drr_ap_final_norm)
    ncc_lp_final = ncc(fixed_lp_norm, drr_lp_final_norm)
    
    print(f"\nFinal NCC scores:")
    print(f"AP view: {ncc_ap_final:.6f}")
    print(f"LP view: {ncc_lp_final:.6f}")
    print(f"Average: {(ncc_ap_final + ncc_lp_final) / 2:.6f}")

if __name__ == "__main__":
    main()
