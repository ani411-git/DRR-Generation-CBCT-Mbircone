import os
import numpy as np
import SimpleITK as sitk
import pydicom
import pydicom.uid
from pydicom.uid import generate_uid
import datetime
from scipy.linalg import rq
import math
import sys

from mbircone.cone3D import project as mbircone_project
from PIL import Image

def load_ct(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not series_files:
        raise FileNotFoundError(f"No DICOM series found in directory: {dicom_dir}")
    reader.SetFileNames(series_files)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    sitk_spacing = image.GetSpacing()
    origin = image.GetOrigin()

    volume_spacing_zyx = (sitk_spacing[2], sitk_spacing[1], sitk_spacing[0])

    metadata = pydicom.dcmread(series_files[0])

    return array, volume_spacing_zyx, origin, metadata


def safe_homogeneous_to_euclidean(h_coords):
    if abs(h_coords[3]) < 1e-8:
        raise ValueError("Invalid homogeneous coordinates: w component too close to zero")
    return h_coords[:3] / h_coords[3]

def validate_rotation_matrix(R):
    I = np.eye(3)
    if not np.allclose(R @ R.T, I, rtol=1e-6, atol=1e-6):
        return False
    
    if not np.isclose(np.linalg.det(R), 1.0, rtol=1e-6):
        return False
    
    return True

def validate_geometry(SID, SAD, magnification):
    if SID <= SAD:
        raise ValueError(f"Invalid geometry: SID ({SID:.2f}) must be greater than SAD ({SAD:.2f})")
    
    expected_mag = SID/SAD
    if not np.isclose(magnification, expected_mag, rtol=1e-3):
        raise ValueError(f"Inconsistent geometry: magnification {magnification:.3f} != SID/SAD {expected_mag:.3f}")

def parse_projection_matrix(file_path, choice='both'):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Projection matrix file not found: {file_path}")
        
    all_numbers_str = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                raise ValueError("Projection matrix file is empty.")
            
            first_line = lines[0].strip()
            all_numbers_str.extend(first_line.split())

            if len(lines) > 1:
                second_line = lines[1].strip()
                all_numbers_str.extend(second_line.split())
            else:
                raise ValueError("Projection matrix file must contain at least two lines for two matrices.")

        nums = np.array(all_numbers_str, dtype=float)

    except ValueError as e:
        raise ValueError(f"Error converting data to float or invalid file format: {e}. Check file content for non-numeric values or incorrect structure.")
    except Exception as e:
        raise ValueError(f"Error reading projection matrix file: {e}")

    if len(nums) != 24:
        raise ValueError(f"Projection matrix file must contain exactly 24 numbers (two 3x4 matrices), but found {len(nums)}. Check the number format or extra characters.")

    if choice == 'both':
        # Parse both matrices and return both source positions
        P_ap = nums[:12].reshape(3, 4)
        P_lp = nums[12:].reshape(3, 4)
        
        # Calculate AP source position
        _, s_ap, Vt_ap = np.linalg.svd(P_ap)
        C_ap_homogeneous = Vt_ap[-1]
        source_ap = safe_homogeneous_to_euclidean(C_ap_homogeneous)
        
        # Calculate LP source position
        _, s_lp, Vt_lp = np.linalg.svd(P_lp)
        C_lp_homogeneous = Vt_lp[-1]
        source_lp = safe_homogeneous_to_euclidean(C_lp_homogeneous)
        
        print(f"Calculated AP source position: {source_ap}")
        print(f"Calculated LP source position: {source_lp}")
        
        return source_ap, source_lp
    
    else:
        # Original single matrix parsing logic
        P_matrix_data = nums[:12] if choice == '1' else nums[12:]
        P = P_matrix_data.reshape(3, 4)

        if np.linalg.cond(P) > 1e10:
            print(f"Warning: Projection matrix is poorly conditioned (condition number: {np.linalg.cond(P):.2e})")

        try:
            _, s, Vt = np.linalg.svd(P)
            print("P matrix used:\n", P)
            print("Singular values (s):\n", s)

            # Use relative threshold based on largest singular value
            threshold = np.max(s) * 1e-6  # Increased threshold
            small_singular_values = np.abs(s) < threshold
            num_small_singular_values = np.sum(small_singular_values)

            if num_small_singular_values == 0:
                # If no small singular values found, use the smallest one
                print(f"No singular values below threshold {threshold:.2e}. Using last singular vector as null space.")
            elif num_small_singular_values > 1:
                raise ValueError(f"Found {num_small_singular_values} singular values below {threshold:.2e}. Expected 1. Matrix may be degenerate.")

            # Always use the last right singular vector as the null space
            C_homogeneous = Vt[-1]
            C = safe_homogeneous_to_euclidean(C_homogeneous)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"SVD computation failed: {e}. This usually means the matrix is degenerate.")
        except Exception as e:
            raise ValueError(f"Error during SVD or camera center extraction: {e}")

        M = P[:, :3]
        try:
            K, R = rq(M)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"RQ decomposition failed: {e}")

        T = np.diag(np.sign(np.diag(K)))
        K = K @ T
        R = T @ R

        if not validate_rotation_matrix(R):
            raise ValueError("Invalid rotation matrix R after RQ decomposition")

        SAD = np.linalg.norm(C)
        if SAD <= 0:
            raise ValueError(f"Invalid SAD value: {SAD}")

        if not np.isclose(K[0,0], K[1,1], rtol=0.1):
            print(f"Warning: Focal lengths differ significantly: {K[0,0]:.2f} vs {K[1,1]:.2f}")
        
        SID = (abs(K[0,0]) + abs(K[1,1])) / 2
        if SID <= 0:
            raise ValueError(f"Invalid SID value: {SID}")

        if SID <= SAD:
            raise ValueError(f"Invalid geometry: SID ({SID:.2f}) must be greater than SAD ({SAD:.2f})")

        # Now that we have SID, calculate detector parameters
        det_size_x = int(2 * K[0, 2])  # Principal point x * 2 gives approximate detector width
        det_size_y = int(2 * K[1, 2])  # Principal point y * 2 gives approximate detector height
        det_spacing_x = SID / K[0, 0]  # Pixel spacing from focal length
        det_spacing_y = SID / K[1, 1]  # Pixel spacing from focal length

        print("\n--- Detector Parameters from P-matrix ---")
        print(f"Detector size (pixels): {det_size_x} x {det_size_y}")
        print(f"Pixel spacing (mm): {det_spacing_x:.4f} x {det_spacing_y:.4f}")
        print("----------------------------------------\n")

        direction = -R[2, :]
        direction = direction / np.linalg.norm(direction)

        # Enhanced principal point validation
        print("\n--- Principal Point Analysis ---")
        cx, cy = K[0,2], K[1,2]
        print(f"Principal point (cx, cy): ({cx:.2f}, {cy:.2f})")
        print(f"Detector bounds: x=[0, {det_size_x}], y=[0, {det_size_y}]")
        
        # Check X bounds
        if cx < 0:
            print(f"WARNING: Principal point X is {abs(cx):.2f} pixels LEFT of detector edge")
        elif cx > det_size_x:
            print(f"WARNING: Principal point X is {cx - det_size_x:.2f} pixels RIGHT of detector edge")
        else:
            print(f"OK: Principal point X at {(cx/det_size_x)*100:.1f}% of detector width")
        
        # Check Y bounds
        if cy < 0:
            print(f"WARNING: Principal point Y is {abs(cy):.2f} pixels ABOVE detector edge")
        elif cy > det_size_y:
            print(f"WARNING: Principal point Y is {cy - det_size_y:.2f} pixels BELOW detector edge")
        else:
            print(f"OK: Principal point Y at {(cy/det_size_y)*100:.1f}% of detector height")
        
        # Check if principal point is near center
        center_x, center_y = det_size_x/2, det_size_y/2
        dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        print(f"Distance from detector center: {dist_from_center:.2f} pixels")
        print("----------------------------------------\n")

        print("\n--- Projection Matrix Validation Results ---")
        print(f"Matrix conditioning number: {np.linalg.cond(P):.2e}")
        print(f"Focal length ratio (K[0,0]/K[1,1]): {K[0,0]/K[1,1]:.3f}")
        print(f"SAD: {SAD:.2f} mm")
        print(f"SID: {SID:.2f} mm")
        print(f"Magnification (SID/SAD): {SID/SAD:.3f}")
        print("Direction vector (normalized):", direction)
        print("----------------------------------------\n")

        return direction, SAD, SID, C, R, det_size_x, det_size_y, det_spacing_x, det_spacing_y, K


def estimate_angle_from_direction(direction):
    """
    Estimate rotation angle from direction vector.
    For AP view: Calculate angle in x-y plane
    For LP view: Calculate angle in y-z plane
    """
    x, y, z = direction
    # Return both angles for AP and LP views
    xy_angle = np.arctan2(x, y)  # For AP view
    yz_angle = np.arctan2(z, y)  # For LP view
    return xy_angle, yz_angle


def generate_drr_mbircone(volume, volume_spacing, SAD, SID, det_size_x, det_size_y, det_spacing_x, det_spacing_y, rotation_angle_deg=0):
    if volume.ndim != 3:
        raise ValueError("Volume must be 3D")
    if not all(s > 0 for s in volume_spacing):
        raise ValueError("All spacing values must be positive")
    
    try:
        validate_geometry(SID, SAD, SID/SAD)
    except ValueError as e:
        raise ValueError(f"Geometry validation failed: {e}")

    volume_float = volume.astype(np.float32)
    angles = np.array([np.radians(rotation_angle_deg)], dtype=np.float32)

    if det_size_x <= 0 or det_size_y <= 0:
        raise ValueError("Detector dimensions must be positive")
    if det_spacing_x <= 0 or det_spacing_y <= 0:
        raise ValueError("Detector spacing must be positive")

    magnification = SID / SAD

    drr_raw_array = mbircone_project(
        image=volume_float,
        angles=angles,
        num_det_rows=det_size_y,
        num_det_channels=det_size_x,
        dist_source_detector=SID,
        magnification=magnification,
        delta_det_row=det_spacing_y,
        delta_det_channel=det_spacing_x,
        delta_pixel_image=volume_spacing[1],
        verbose=0
    )[0]

    drr_array = drr_raw_array.astype(np.float32)

    # Enhanced post-processing to match X-ray appearance
    min_val = np.min(drr_array)
    max_val = np.max(drr_array)
    
    if max_val > min_val:
        # 1. Initial normalization
        drr_array = (drr_array - min_val) / (max_val - min_val)
        
        # 2. Apply stronger gamma correction for bone emphasis
        gamma = 0.5  # Reduced from 0.7 to brighten image more
        drr_array = np.power(drr_array, gamma)
        
        # 3. Enhanced contrast using adaptive histogram equalization
        p1 = np.percentile(drr_array, 1)   # Dark point
        p99 = np.percentile(drr_array, 99)  # Bright point
        drr_array = np.clip(drr_array, p1, p99)
        drr_array = (drr_array - p1) / (p99 - p1)
        
        # 4. Create circular mask
        y, x = np.ogrid[:drr_array.shape[0], :drr_array.shape[1]]
        center_y, center_x = drr_array.shape[0] / 2, drr_array.shape[1] / 2
        radius = min(center_y, center_x)
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        circular_mask = dist_from_center <= radius
        
        # 5. Apply edge darkening
        edge_weight = 1 - (dist_from_center / (radius * 1.2))**2
        edge_weight = np.clip(edge_weight, 0, 1)
        drr_array = drr_array * edge_weight
        
        # 6. Apply circular mask
        drr_array = drr_array * circular_mask
        
        # 7. Final contrast enhancement
        drr_array = np.clip(drr_array * 1.2, 0, 1)  # Boost contrast
        
        # 8. Scale to 8-bit range
        drr_array = drr_array * 255.0
    else:
        drr_array = np.zeros_like(drr_array)

    # Ensure output is properly clipped
    drr_array = np.clip(drr_array, 0, 255)
    
    return drr_array.astype(np.uint8)


def get_next_sequence_number(output_path, label):
    """
    Get the next available sequence number for the given label.
    Checks both .dcm and .png files.
    """
    # Check both DICOM and PNG files
    existing_files = [f for f in os.listdir(output_path) 
                     if f.startswith(label) and (f.endswith('.dcm') or f.endswith('.png'))]
    if not existing_files:
        return 1
    
    # Extract numbers from existing files
    numbers = []
    for f in existing_files:
        try:
            # Remove label and extension (.dcm or .png)
            num = int(''.join(filter(str.isdigit, f)))
            numbers.append(num)
        except ValueError:
            continue
    return max(numbers, default=0) + 1

def save_drr_as_dicom(drr_array, drr_pixel_spacing, output_path, filename, metadata=None, label="DRR"):
    """Save DRR as DICOM file with given filename"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ds = pydicom.Dataset()
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    ds.file_meta.ImplementationVersionName = "PYDICOM " + pydicom.__version__


    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.SOPInstanceUID = generate_uid()

    if metadata:
        ds.PatientName = metadata.get('PatientName', 'Unknown Patient')
        ds.PatientID = metadata.get('PatientID', 'Unknown ID')
        ds.PatientBirthDate = metadata.get('PatientBirthDate', '')
        ds.PatientSex = metadata.get('PatientSex', '')

        ds.StudyInstanceUID = metadata.get('StudyInstanceUID', generate_uid())
        ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
        ds.AccessionNumber = metadata.get('AccessionNumber', '')
        ds.ReferringPhysicianName = metadata.get('ReferringPhysicianName', '')
        ds.StudyID = metadata.get('StudyID', '1')
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = metadata.get('SeriesNumber', '99')
        ds.SeriesDescription = f"{label} DRR"
    else:
        ds.PatientName = "Unknown Patient"
        ds.PatientID = "Unknown ID"
        ds.StudyInstanceUID = generate_uid()
        ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = "99"
        ds.SeriesDescription = f"{label} DRR"


    ds.Modality = "OT"
    ds.InstanceNumber = 1
    ds.ImageComments = f"Generated {label} DRR from CT volume"
    ds.DateOfSecondaryCapture = datetime.datetime.now().strftime('%Y%m%d')
    ds.TimeOfSecondaryCapture = datetime.datetime.now().strftime('%H%M%S')


    ds.Rows, ds.Columns = drr_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0

    ds.PixelSpacing = [str(drr_pixel_spacing[0]), str(drr_pixel_spacing[1])]

    ds.PixelData = drr_array.tobytes()

    ds.SpecificCharacterSet = "ISO_IR 100"

    # Adjust DICOM window settings for better default display
    ds.WindowCenter = 127
    ds.WindowWidth = 255
    ds.PhotometricInterpretation = "MONOCHROME2"

    full_path = os.path.join(output_path, filename)
    try:
        pydicom.filewriter.dcmwrite(full_path, ds, write_like_original=False)
        print(f"DICOM DRR saved to: {full_path}")
    except Exception as e:
        print(f"Error saving DICOM file {full_path}: {e}")

def save_drr_as_png(drr_array, output_path, filename):
    """Save DRR as PNG file with given filename"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    full_path = os.path.join(output_path, filename)
    try:
        img = Image.fromarray(drr_array)
        img.save(full_path)
        print(f"PNG image saved to: {full_path}")
    except Exception as e:
        print(f"Error saving PNG file {full_path}: {e}")

def main_get_pixel_spacing():
    """
    Main function to parse a specific projection matrix file and print detector spacings.
    """
    matrix_file_path = "/Users/aravindr/Downloads/P_MATRIX_FILE.txt"
    view_idx_to_parse = '1' # Assuming you want to parse the first matrix in the file for the X-ray

    print(f"Attempting to parse matrix {view_idx_to_parse} from '{matrix_file_path}' to get detector pixel spacing.\n")

    try:
        # The parse_projection_matrix in mydrr1.py returns many values.
        # We need to capture all of them as defined by that function.
        # Based on its usage in moreexp.py, the return order is:
        # direction, SAD, SID, source_pos, R_matrix, 
        # det_size_x, det_size_y, det_spacing_x, det_spacing_y, K_matrix
        
        # We need to know the exact signature or safely call it.
        # For now, let's assume we can adapt or it prints enough.
        # A robust way is to call it and get all results.
        
        parsed_params = parse_projection_matrix(matrix_file_path, view_idx_to_parse)
        
        # Assuming the order of return values from parse_projection_matrix in mydrr1.py is:
        # (direction, SAD, SID, source_pos, R_matrix, det_size_x, det_size_y, 
        #  det_spacing_x, det_spacing_y, K_matrix)
        # This matches the expected unpacking in moreexp.py
        
        if len(parsed_params) == 10:
            det_spacing_x = parsed_params[7]
            det_spacing_y = parsed_params[8]
            sid_val = parsed_params[2]
            k_matrix = parsed_params[9]
            fx = k_matrix[0,0]
            fy = k_matrix[1,1]


            print("\n--- Extracted Parameters for X-ray System ---")
            print(f"  Calculated SID: {sid_val}")
            print(f"  Intrinsic Matrix K:\n{k_matrix}")
            print(f"  Focal length fx (pixels): {fx}")
            print(f"  Focal length fy (pixels): {fy}")
            print(f"  Detector Pixel Spacing (X): {det_spacing_x} (e.g., mm/pixel)")
            print(f"  Detector Pixel Spacing (Y): {det_spacing_y} (e.g., mm/pixel)")
            print("---------------------------------------------")
            print("These detector spacings are the physical pixel size of your X-ray detector,")
            print("assuming the P_MATRIX_FILE.txt accurately represents your X-ray system.")

        else:
            print("Error: The parse_projection_matrix function in mydrr1.py did not return the expected number of parameters (10).")
            print(f"Received {len(parsed_params)} parameters: {parsed_params}")


    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error during parsing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Comment out or remove other main functions if they exist, or rename this one
    # and call it specifically. For now, this will be the one that runs.
    main_get_pixel_spacing()