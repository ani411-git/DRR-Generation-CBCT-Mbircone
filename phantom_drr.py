import os
import numpy as np
from mbircone.cone3D import project as mbircone_project
from mbircone.vtkig import display_ct_volume
import vtk
import sys
from scipy.spatial.transform import Rotation
import datetime
import pydicom
import matplotlib.pyplot as plt
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import rq
from PIL import Image


def pad_image_to_square(image):
    height, width = image.shape
    max_dim = max(height, width)
    square_image = np.zeros((max_dim, max_dim), dtype=image.dtype)
    pad_height = (max_dim - height) // 2
    pad_width = (max_dim - width) // 2
    square_image[pad_height:pad_height + height, pad_width:pad_width + width] = image
    
    return square_image, pad_height, pad_width

def create_image_viewer(image_data, window_title, label_text=None, output_dir=None, metadata=None, pixel_spacing_yx=None, view_name=None):
    image_data, pad_height, pad_width = pad_image_to_square(image_data)
    image_array = np.copy(image_data)
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(image_data.shape[1], image_data.shape[0], 1)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            vtk_image.SetScalarComponentFromFloat(j, i, 0, 0, float(image_data[i, j]))
    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(vtk_image)
    image_property = image_actor.GetProperty()
    image_property.SetColorWindow(255)
    image_property.SetColorLevel(127.5)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(image_actor)
    renderer.ResetCamera()
    if label_text:
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(label_text)
        text_prop = text_actor.GetTextProperty()
        text_prop.SetFontSize(24)
        text_prop.SetColor(1.0, 1.0, 1.0)
        text_actor.SetDisplayPosition(10, 10)
        renderer.AddActor2D(text_actor)
    save_button = vtk.vtkTextActor()
    save_button.SetInput("Save")
    button_prop = save_button.GetTextProperty()
    button_prop.SetFontSize(20)
    button_prop.SetColor(0.0, 1.0, 0.0)
    button_prop.SetBackgroundColor(0.2, 0.2, 0.2)
    button_prop.SetBackgroundOpacity(0.8)
    save_button_pos = (10, 50)
    save_button.SetDisplayPosition(save_button_pos[0], save_button_pos[1])
    renderer.AddActor2D(save_button)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName(window_title)
    render_window.SetSize(600, 600)
    render_interactor = vtk.vtkRenderWindowInteractor()
    render_interactor.SetRenderWindow(render_window)
    class ButtonStyle(vtk.vtkInteractorStyleImage):
        def __init__(self):
            self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
            self.AddObserver("MouseWheelForwardEvent", self.mouse_wheel_forward_event)
            self.AddObserver("MouseWheelBackwardEvent", self.mouse_wheel_backward_event)
            self.AddObserver("KeyPressEvent", self.key_press_event)
            self.AddObserver("PinchEvent", self.pinch_event)
            self.AddObserver("KeyReleaseEvent", self.key_release_event)
            self.save_button = save_button
            self.save_button_pos = save_button_pos
            self.text_actor = text_actor
            self.renderer = renderer
            self.render_window = render_window
            self.window_title = window_title
            self.image_actor = image_actor
            self.vtk_image = vtk_image
            self.image_array = image_array
            self.output_dir = output_dir
            self.metadata = metadata
            self.pixel_spacing_yx = pixel_spacing_yx
            self.view_name = view_name
            self.brightness_mode = False
            self.zoom_factor = 1.0
            self.vertical_shift_amount = 0
            self.horizontal_shift_amount = 0
            self.pad_height = pad_height
            self.pad_width = pad_width
            self.ap_window = None
            self.lp_window = None
            self.ap_zoom = 1.0
            self.ap_vertical_shift = 0
            self.ap_horizontal_shift = 0
            self.lp_zoom = 1.0
            self.lp_vertical_shift = 0
            self.lp_horizontal_shift = 0
            self.circular_mask = self.create_circular_mask()
            self.image_array = self.apply_circular_mask(self.image_array, self.circular_mask)
            self.update_vtk_image()
        def create_circular_mask(self):
            height, width = self.image_array.shape
            center_y, center_x = height / 2.0, width / 2.0
            radius = min(height, width) / 2.0 - 10
            Y, X = np.ogrid[:height, :width]
            distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            mask = (distance_from_center <= radius).astype(np.uint8)
            return mask
        def apply_circular_mask(self, image, mask):
            masked_image = image * mask
            return masked_image
        def update_vtk_image(self):
            self.image_array = self.apply_circular_mask(self.image_array, self.circular_mask)
            for i in range(self.image_array.shape[0]):
                for j in range(self.image_array.shape[1]):
                    self.vtk_image.SetScalarComponentFromFloat(j, i, 0, 0, float(self.image_array[i, j]))
            self.vtk_image.Modified()
            self.image_actor.SetInputData(self.vtk_image)
            self.render_window.Render()
        def key_press_event(self, obj, event):
            key = self.GetInteractor().GetKeySym()
            print(f"\nKey pressed: '{key}'")
            if key == 'b':
                self.brightness_mode = not self.brightness_mode
                state = "ON" if self.brightness_mode else "OFF"
                print(f"Brightness adjustment mode: {state} (trackpad scroll to adjust)")
            elif key in ['plus', 'equal', 'Plus', 'Equal', '+']:
                self.zoom_factor *= 1.1
                camera = self.renderer.GetActiveCamera()
                camera.Zoom(self.zoom_factor)
                print(f"\nDEBUG: Zoom in (key) - View: {self.view_name}, Zoom factor: {self.zoom_factor:.2f}")
                self.render_window.Render()
            elif key in ['minus', 'Minus', '-']:
                self.zoom_factor /= 1.1
                camera = self.renderer.GetActiveCamera()
                camera.Zoom(self.zoom_factor)
                print(f"\nDEBUG: Zoom out (key) - View: {self.view_name}, Zoom factor: {self.zoom_factor:.2f}")
                self.render_window.Render()
            elif key in ['Up', 'uparrow']:
                shift_step = 5
                self.vertical_shift_amount += shift_step
                self.image_array = np.roll(self.image_array, -shift_step, axis=0)
                self.image_array[-shift_step:, :] = 0
                self.update_vtk_image()
                print(f"\nDEBUG: Shifted image content up - View: {self.view_name}, Total vertical shift: {self.vertical_shift_amount} rows")
            elif key in ['Down', 'downarrow']:
                shift_step = 5
                self.vertical_shift_amount -= shift_step
                self.image_array = np.roll(self.image_array, shift_step, axis=0)
                self.image_array[:shift_step, :] = 0
                self.update_vtk_image()
                print(f"\nDEBUG: Shifted image content down - View: {self.view_name}, Total vertical shift: {self.vertical_shift_amount} rows")
            elif key in ['Left', 'leftarrow']:
                shift_step = 5
                self.horizontal_shift_amount -= shift_step
                self.image_array = np.roll(self.image_array, -shift_step, axis=1)
                self.image_array[:, -shift_step:] = 0
                self.update_vtk_image()
                print(f"\nDEBUG: Shifted image content left - View: {self.view_name}, Total horizontal shift: {self.horizontal_shift_amount} columns")
            elif key in ['Right', 'rightarrow']:
                shift_step = 5
                self.horizontal_shift_amount += shift_step
                self.image_array = np.roll(self.image_array, shift_step, axis=1)
                self.image_array[:, :shift_step] = 0
                self.update_vtk_image()
                print(f"\nDEBUG: Shifted image content right - View: {self.view_name}, Total horizontal shift: {self.horizontal_shift_amount} columns")
        def key_release_event(self, obj, event):
            key = self.GetInteractor().GetKeySym()
            print(f"Key released: '{key}'")
        def mouse_wheel_forward_event(self, obj, event):
            if self.brightness_mode:
                current_level = self.image_actor.GetProperty().GetColorLevel()
                self.image_actor.GetProperty().SetColorLevel(current_level + 10)
                self.render_window.Render()
            else:
                self.zoom_factor *= 1.1
                camera = self.renderer.GetActiveCamera()
                camera.Zoom(self.zoom_factor)
                print(f"\nDEBUG: Zoom in (scroll) - View: {self.view_name}, Zoom factor: {self.zoom_factor:.2f}")
                self.render_window.Render()
        def mouse_wheel_backward_event(self, obj, event):
            if self.brightness_mode:
                current_level = self.image_actor.GetProperty().GetColorLevel()
                self.image_actor.GetProperty().SetColorLevel(current_level - 10)
                self.render_window.Render()
            else:
                self.zoom_factor /= 1.1
                camera = self.renderer.GetActiveCamera()
                camera.Zoom(self.zoom_factor)
                print(f"\nDEBUG: Zoom out (scroll) - View: {self.view_name}, Zoom factor: {self.zoom_factor:.2f}")
                self.render_window.Render()
        def pinch_event(self, obj, event):
            try:
                scale = self.GetInteractor().GetScale()
                if scale != 1.0:
                    self.zoom_factor *= scale
                    camera = self.renderer.GetActiveCamera()
                    camera.Zoom(self.zoom_factor)
                    print(f"\nPinch-to-zoom - Scale: {scale:.2f}, Zoom factor: {self.zoom_factor:.2f}")
                    self.render_window.Render()
                else:
                    print("\nPinch event received, but no scale change detected.")
            except AttributeError:
                print("\nPinch event handling not fully supported in this VTK version.")
        def left_button_press_event(self, obj, event):
            click_pos = self.GetInteractor().GetEventPosition()
            print(f"\nClick detected at position: {click_pos}")
            save_button_pos = self.save_button_pos
            print(f"Save button position: {save_button_pos}")
            self.save_button.GetTextProperty().SetFontSize(20)
            size = [0, 0]
            self.save_button.GetSize(self.renderer, size)
            save_button_width, save_button_height = size[0], size[1]
            if save_button_width == 0 or save_button_height == 0:
                print("GetSize returned 0, using fallback size (50x30)")
                save_button_width, save_button_height = 50, 30
            else:
                print(f"Save button size: {save_button_width}x{save_button_height}")
            if (save_button_pos[0] <= click_pos[0] <= save_button_pos[0] + save_button_width and
                save_button_pos[1] <= click_pos[1] <= save_button_pos[1] + save_button_height):
                print(f"\nSave button clicked in {self.window_title} window")
                self.save_button.GetTextProperty().SetColor(1.0, 1.0, 0.0)
                self.GetInteractor().GetRenderWindow().Render()
                self.save_drr()
                self.save_button.GetTextProperty().SetColor(0.0, 1.0, 0.0)
                self.GetInteractor().GetRenderWindow().Render()
            else:
                print(f"Click outside save button bounds (x: {save_button_pos[0]} to {save_button_pos[0] + save_button_width}, "
                      f"y: {save_button_pos[1]} to {save_button_pos[1] + save_button_height})")
            self.OnLeftButtonDown()
        def save_drr(self):
            print("\nDEBUG: save_drr method called")
            print(f"DEBUG: Current view: {self.view_name}")
            print(f"DEBUG: Current zoom factor: {self.zoom_factor}")
            print(f"DEBUG: Current vertical shift: {self.vertical_shift_amount}")
            print(f"DEBUG: Current horizontal shift: {self.horizontal_shift_amount}")
            if self.view_name == "AP":
                self.ap_zoom = self.zoom_factor
                self.ap_vertical_shift = self.vertical_shift_amount
                self.ap_horizontal_shift = self.horizontal_shift_amount
                print("\nDEBUG: Updated AP View Adjustments:")
                print(f"Zoom: {self.ap_zoom}")
                print(f"Vertical Shift: {self.ap_vertical_shift}")
                print(f"Horizontal Shift: {self.ap_horizontal_shift}")
            else:
                self.lp_zoom = self.zoom_factor
                self.lp_vertical_shift = self.vertical_shift_amount
                self.lp_horizontal_shift = self.horizontal_shift_amount
                print("\nDEBUG: Updated LP View Adjustments:")
                print(f"Zoom: {self.lp_zoom}")
                print(f"Vertical Shift: {self.lp_vertical_shift}")
                print(f"Horizontal Shift: {self.lp_horizontal_shift}")
            seq_num = get_next_sequence_number(self.output_dir, self.view_name)
            dicom_filename = f"{self.view_name}_{seq_num}.dcm"
            png_filename = f"{self.view_name}_{seq_num}.png"
            save_drr_as_dicom(self.image_array, self.pixel_spacing_yx, self.output_dir, dicom_filename, self.metadata, self.view_name)
            save_drr_as_png(self.image_array, self.output_dir, png_filename)
        def set_window_references(self, ap_window, lp_window):
            self.ap_window = ap_window
            self.lp_window = lp_window
    style = ButtonStyle()
    style.SetInteractionModeToImage2D()
    render_interactor.SetInteractorStyle(style)
    return render_window, render_interactor, style

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
    expected_mag = SID / SAD
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
                raise ValueError("Projection matrix file is empty")
            first_line = lines[0].strip()
            all_numbers_str.extend(first_line.split())
            if len(lines) > 1:
                second_line = lines[1].strip()
                all_numbers_str.extend(second_line.split())
            else:
                raise ValueError("Projection matrix file must contain at least two lines for two matrices")
        nums = np.array(all_numbers_str, dtype=float)
    except ValueError as e:
        raise ValueError(f"Error converting data to float or invalid file format: {e}")
    except Exception as e:
        raise ValueError(f"Error reading projection matrix file: {e}")
    if len(nums) != 24:
        raise ValueError(f"Projection matrix file must contain exactly 24 numbers (two 3x4 matrices), but found {len(nums)}")
    if choice == 'both':
        P_ap = nums[:12].reshape(3, 4)
        P_lp = nums[12:].reshape(3, 4)
        _, s_ap, Vt_ap = np.linalg.svd(P_ap)
        C_ap_homogeneous = Vt_ap[-1]
        source_ap = safe_homogeneous_to_euclidean(C_ap_homogeneous)
        _, s_lp, Vt_lp = np.linalg.svd(P_lp)
        C_lp_homogeneous = Vt_lp[-1]
        source_lp = safe_homogeneous_to_euclidean(C_lp_homogeneous)
        print(f"Calculated AP source position: {source_ap}")
        print(f"Calculated LP source position: {source_lp}")
        return source_ap, source_lp
    else:
        P_matrix_data = nums[:12] if choice == '1' else nums[12:]
        P = P_matrix_data.reshape(3, 4)
        if np.linalg.cond(P) > 1e10:
            print(f"Warning: Projection matrix is poorly conditioned (condition number: {np.linalg.cond(P):.2e})")
        try:
            _, s, Vt = np.linalg.svd(P)
            print("P matrix used:\n", P)
            print("Singular values (s):\n", s)
            threshold = np.max(s) * 1e-6
            small_singular_values = np.abs(s) < threshold
            num_small_singular_values = np.sum(small_singular_values)
            if num_small_singular_values == 0:
                print(f"No singular values below threshold {threshold:.2e}. Using last singular vector as null space")
            elif num_small_singular_values > 1:
                raise ValueError(f"Found {num_small_singular_values} singular values below {threshold:.2e}. Expected 1")
            C_homogeneous = Vt[-1]
            C = safe_homogeneous_to_euclidean(C_homogeneous)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"SVD computation failed: {e}")
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
        det_size_x = int(2 * K[0, 2])
        det_size_y = int(2 * K[1, 2])
        det_spacing_x = SID / K[0, 0]
        det_spacing_y = SID / K[1, 1]
        print("\n--- Detector Parameters from P-matrix ---")
        print(f"Detector size (pixels): {det_size_x} x {det_size_y}")
        print(f"Pixel spacing (mm): {det_spacing_x:.4f} x {det_spacing_y:.4f}")
        print("----------------------------------------\n")
        direction = [0,0,1]
        direction = direction / np.linalg.norm(direction)
        print("\n--- Principal Point Analysis ---")
        # Hardcode principal point to center of hardcoded detector
        cy = 1052 / 2  # det_size_y / 2
        cx = 1024 / 2  # det_size_x / 2
        cx, cy = K[0,2], K[1,2]
        print(f"Principal point (cx, cy): ({cx:.2f}, {cy:.2f})")
        print(f"Detector bounds: x=[0, {det_size_x}], y=[0, {det_size_y}]")
        if cx < 0:
            print(f"WARNING: Principal point X is {abs(cx):.2f} pixels LEFT of detector edge")
        elif cx > det_size_x:
            print(f"WARNING: Principal point X is {cx - det_size_x:.2f} pixels RIGHT of detector edge")
        else:
            print(f"OK: Principal point X at {(cx/det_size_x)*100:.1f}% of detector width")
        if cy < 0:
            print(f"WARNING: Principal point Y is {abs(cy):.2f} pixels ABOVE detector edge")
        elif cy > det_size_y:
            print(f"WARNING: Principal point Y is {cy - det_size_y:.2f} pixels BELOW detector edge")
        else:
            print(f"OK: Principal point Y at {(cy/det_size_y)*100:.1f}% of detector height")
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
    x, y, z = direction
    xy_angle = np.arctan2(x, y)
    yz_angle = np.arctan2(z, y)
    xy_angle_deg = np.degrees(xy_angle)
    if xy_angle_deg < 0:
        xy_angle_deg += 360
    return xy_angle_deg, np.degrees(yz_angle)

def decompose_projection_matrix(P, pixel_spacing=0.741):
    _, s, Vt = np.linalg.svd(P)
    C_homogeneous = Vt[-1]
    source_position = safe_homogeneous_to_euclidean(C_homogeneous)
    M = P[:, :3]
    K, R = rq(M)
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    if not validate_rotation_matrix(R):
        raise ValueError("Invalid rotation matrix from decomposition")
    t_homogeneous = P[:, 3]
    translation_norm = safe_homogeneous_to_euclidean(np.append(t_homogeneous, 1))
    f_x = K[0, 0]
    f_y = K[1, 1]
    f_x_mm = f_x * pixel_spacing
    f_y_mm = f_y * pixel_spacing
    SDD = (f_x_mm + f_y_mm) / 2
    sourcetiso = np.linalg.norm(source_position)
    z_axis_camera = np.array([0, 0, 1])
    z_axis_world = R.T @ z_axis_camera
    detector_center = source_position + z_axis_world * SDD
    return source_position, SDD, detector_center, R, translation_norm, K, sourcetiso

def get_projection_matrix(euler_angles_deg, translation_vec, K):
    angles_rad = np.deg2rad(euler_angles_deg)
    R = Rotation.from_euler('xyz', angles_rad, degrees=False).as_matrix()
    t = np.array(translation_vec).reshape(3, 1)
    Rt = np.hstack((R, t))
    P = K @ Rt
    return P

def get_ct_isocenter(volume_hu, ct_spacing_zyx, origin):
    return np.array([126.79, 109.13, 143.11])

def numpy_to_vtk_image_data(numpy_array):
    if numpy_array.dtype != np.uint8:
        raise ValueError("Input NumPy array must be of type uint8 for numpy_to_vtk_image_data.")
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(numpy_array.shape[1], numpy_array.shape[0], 1)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    for i in range(numpy_array.shape[0]):
        for j in range(numpy_array.shape[1]):
            vtk_image.SetScalarComponentFromFloat(j, i, 0, 0, float(numpy_array[i, j]))
    return vtk_image

def save_vtk_image_as_png(vtk_image_data, filename):
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_image_data)
    try:
        writer.Write()
        print(f"Saved DRR as PNG: {filename}")
    except Exception as e:
        print(f"Error saving PNG {filename}: {e}")

def get_next_sequence_number(output_path, label):
    existing_files = [f for f in os.listdir(output_path) if f.startswith(label) and (f.endswith('.dcm') or f.endswith('.png'))]
    if not existing_files:
        return 1
    numbers = []
    for f in existing_files:
        try:
            num = int(''.join(filter(str.isdigit, f)))
            numbers.append(num)
        except ValueError:
            continue
    return max(numbers, default=0) + 1

def save_drr_as_dicom(drr_array, drr_pixel_spacing, output_path, filename, metadata=None, label="DRR"):
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
    ds.WindowCenter = 127
    ds.WindowWidth = 255
    ds.PixelData = drr_array.tobytes()
    ds.SpecificCharacterSet = "ISO_IR 100"
    full_path = os.path.join(output_path, filename)
    try:
        pydicom.filewriter.dcmwrite(full_path, ds, write_like_original=False)
        print(f"DICOM DRR saved to: {full_path}")
    except Exception as e:
        print(f"Error saving DICOM file {full_path}: {e}")

def save_drr_as_png(drr_array, output_path, filename):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    full_path = os.path.join(output_path, filename)
    try:
        img = Image.fromarray(drr_array)
        img.save(full_path)
        print(f"PNG image saved to: {full_path}")
    except Exception as e:
        print(f"Error saving PNG file {full_path}: {e}")

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

def normalize_drr_for_display(drr):
    min_val = np.min(drr)
    max_val = np.max(drr)
    if max_val > min_val:
        return ((drr - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        return np.full_like(drr, 128 if max_val > 0 else 0, dtype=np.uint8)

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
    min_val = np.min(drr_array)
    max_val = np.max(drr_array)
    if max_val > min_val:
        drr_array = (drr_array - min_val) / (max_val - min_val)
        gamma = 0.7
        drr_array = np.power(drr_array, gamma)
        p1 = np.percentile(drr_array, 1)
        p99 = np.percentile(drr_array, 99)
        drr_array = np.clip(drr_array, p1, p99)
        drr_array = (drr_array - p1) / (p99 - p1)
        y, x = np.ogrid[:drr_array.shape[0], :drr_array.shape[1]]
        center_y, center_x = drr_array.shape[0] / 2, drr_array.shape[1] / 2
        radius = min(center_y, center_x) * 0.9
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        circular_mask = dist_from_center <= radius
        edge_weight = 1 - (dist_from_center / (radius * 1.2))**2
        edge_weight = np.clip(edge_weight, 0.2, 1)
        drr_array = drr_array * edge_weight
        drr_array = drr_array * circular_mask
        drr_array = np.clip(drr_array * 1.5, 0, 1)
        drr_array = drr_array * 255.0
    else:
        drr_array = np.zeros_like(drr_array)
    drr_array = np.clip(drr_array, 0, 255)
    return drr_array.astype(np.uint8)

def visualize_camera_setup(source_ap, source_lp, isocenter, detector_center_ap=None, detector_center_lp=None, direction_ap=None, direction_lp=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*isocenter, color='green', s=80, label='CT Isocenter')
    ax.text(*isocenter, "Isocenter", color='green')
    ax.scatter(*source_ap, color='red', s=60, label='AP Source')
    ax.text(*source_ap, "AP Source", color='red')
    ax.plot([source_ap[0], isocenter[0]], [source_ap[1], isocenter[1]], [source_ap[2], isocenter[2]], 'r--')
    ax.scatter(*source_lp, color='blue', s=60, label='LP Source')
    ax.text(*source_lp, "LP Source", color='blue')
    ax.plot([source_lp[0], isocenter[0]], [source_lp[1], isocenter[1]], [source_lp[2], isocenter[2]], 'b--')
    if detector_center_ap is not None:
        ax.scatter(*detector_center_ap, color='orange', s=60, label='AP Detector Center')
        ax.text(*detector_center_ap, "AP Detector", color='orange')
        ax.plot([source_ap[0], detector_center_ap[0]], [source_ap[1], detector_center_ap[1]], [source_ap[2], detector_center_ap[2]], 'r:')
    if detector_center_lp is not None:
        ax.scatter(*detector_center_lp, color='cyan', s=60, label='LP Detector Center')
        ax.text(*detector_center_lp, "LP Detector", color='cyan')
        ax.plot([source_lp[0], detector_center_lp[0]], [source_lp[1], detector_center_lp[1]], [source_lp[2], detector_center_lp[2]], 'b:')
    if direction_ap is not None:
        arrow_len = 4000
        dir_ap_end = isocenter + arrow_len * direction_ap
        ax.quiver(isocenter[0], isocenter[1], isocenter[2],
                  direction_ap[0], direction_ap[1], direction_ap[2],
                  length=arrow_len, color='magenta', label='AP Direction')
    if direction_lp is not None:
        arrow_len = 4000
        dir_lp_end = isocenter + arrow_len * direction_lp
        ax.quiver(isocenter[0], isocenter[1], isocenter[2],
                  direction_lp[0], direction_lp[1], direction_lp[2],
                  length=arrow_len, color='cyan', label='LP Direction')
    ax.set_title("Source–Isocenter–Detector Configuration", fontsize=14)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=20, azim=45)
    ax.set_xlim(-6000, 1000)
    ax.set_ylim(-6000, 1000)
    ax.set_zlim(0, 200)
    plt.tight_layout()
    plt.show()


def main():
    base_download_dir = os.path.expanduser("~/Downloads")
    dicom_folder = os.path.join(base_download_dir, "A_CT_SCAN_FOLDER")
    output_dir = os.path.join(base_download_dir, "proj output")
    os.makedirs(output_dir, exist_ok=True)
    old_matrix_path = '/Users/aravindr/Downloads/P_MATRIX_FILE.txt'
    output_new_P_matrix_file_path = os.path.join(output_dir, "new_P_matrix.txt")
    magnification_scale_factor_ap = 1.0
    magnification_scale_factor_lp = 1.0
    matrix_choice_lp = '2'
    matrix_choice_ap = '1'
    ap_window, ap_interactor = None, None
    lp_window, lp_interactor = None, None
    print("\nLoading CT volume...")
    try:
        volume_hu, ct_spacing_zyx, origin, metadata = load_ct(dicom_folder)
        print(f"CT volume loaded. Shape: {volume_hu.shape}, Spacing: {ct_spacing_zyx}")
        print(f"CT volume origin: {origin} mm")
        print(f"Initial HU range: [{volume_hu.min():.2f}, {volume_hu.max():.2f}]")
        isocenter_ct_hivtk = get_ct_isocenter(volume_hu, ct_spacing_zyx, np.array(origin))
        print(f"Computed CT isocenter (mm): {isocenter_ct_hivtk}")
    except Exception as e:
        print(f"\nError loading CT volume: {str(e)}")
        return
    print("\nConverting to attenuation coefficients...")
    # Convert from LPS to RAS by flipping the first two spatial axes (Y and X)
    volume_hu_ras = volume_hu[:, ::-1, ::-1]
    # Now convert to attenuation coefficients
    mu_water_at_80kev = 0.02
    volume_attenuation = ((volume_hu_ras + 1000.0) / 1000.0) * mu_water_at_80kev
    volume_attenuation[volume_attenuation < 0] = 0
    print("Conversion complete.")
    P1 = np.array([
        [441.718242, 3523.733037, -2033.378096, 217409.124851],
    [-1160.510253, -1774.851122, -3484.694510, -199449.148331],
    [-0.948070, 0.317982, 0.007080, 745.199857]
        
    ], dtype=np.float64)
    P2 = np.array([[-4298.936264, 801.573106, 504.814267, 758951.144144],
    [-957.239585, -2682.742358, -3351.795366, -201860.747034],
    [-0.207768, -0.806154, 0.554029, 815.235148]
        
    ], dtype=np.float64)
    print("\n--- Decomposing Projection Matrix P1 (LP) ---")
    source_lp, sdd_lp, detector_center_lp, R_lp, translation_norm_lp, K_lp, sourcetiso_lp = decompose_projection_matrix(P1)
    print("\n--- Decomposing Projection Matrix P2 (AP) ---")
    source_ap, sdd_ap, detector_center_ap, R_ap, translation_norm_ap, K_ap, sourcetiso_ap = decompose_projection_matrix(P2)
    rx, ry, rz = 32.3818908, -7.81284325, -169.96967506
    tx, ty, tz = 194.63683, 157.27507, 299.9288
    rot = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    rot_matrix = rot.as_matrix()
    source_ap_new = rot_matrix @ source_ap + np.array([tx, ty, tz])
    detector_center_ap_new = rot_matrix @ detector_center_ap + np.array([tx, ty, tz])
    R_ap_new = rot_matrix @ R_ap
    source_lp_new = rot_matrix @ source_lp + np.array([tx, ty, tz])
    detector_center_lp_new = rot_matrix @ detector_center_lp + np.array([tx, ty, tz])
    R_lp_new = rot_matrix @ R_lp
    print("\n--- New AP C-arm Extrinsic Parameters ---")
    r_ap = Rotation.from_matrix(R_ap_new)
    euler_ap = r_ap.as_euler('xyz', degrees=True)
    t_ap = -R_ap_new @ source_ap_new
    print("Rotation (Euler angles, deg):", euler_ap)
    print("Translation vector:", t_ap)
    print("\n--- New LP C-arm Extrinsic Parameters ---")
    r_lp = Rotation.from_matrix(R_lp_new)
    euler_lp = r_lp.as_euler('xyz', degrees=True)
    t_lp = -R_lp_new @ source_lp_new
    print("Rotation (Euler angles, deg):", euler_lp)
    print("Translation vector:", t_lp)
    P_new_ap = get_projection_matrix(euler_ap, t_ap, K_ap)
    P_new_lp = get_projection_matrix(euler_lp, t_lp, K_lp)
    try:
        with open(output_new_P_matrix_file_path, 'w') as f:
            f.write(' '.join(f"{num:.6f}" for num in P_new_ap.flatten()) + '\n')
            f.write(' '.join(f"{num:.6f}" for num in P_new_lp.flatten()) + '\n')
        print(f"Successfully saved new projection matrices to: {output_new_P_matrix_file_path}")
    except Exception as e:
        print(f"Error saving new projection matrices: {e}")
    try:
        # All downstream code must treat P1 as LP and P2 as AP
        # (No swap needed here, but ensure all usages are correct)
        #
        # When parsing the new projection matrix file:
        direction_lp_new, SAD_lp_new, SID_lp_new, source_lp_new, R_lp_new, det_size_x_lp, det_size_y_lp, det_spacing_x_lp, det_spacing_y_lp, K_lp = parse_projection_matrix(output_new_P_matrix_file_path, matrix_choice_lp) # P1 (LP)
        direction_ap_new, SAD_ap_new, SID_ap_new, source_ap_new, R_ap_new, det_size_x_ap, det_size_y_ap, det_spacing_x_ap, det_spacing_y_ap, K_ap = parse_projection_matrix(output_new_P_matrix_file_path, matrix_choice_ap) # P2 (AP)
        #
        # DRR generation: use LP variables for LP, AP variables for AP
        fov_scale_factor = 1.0
        SID_lp_adjusted = SID_lp_new * magnification_scale_factor_lp
        SID_ap_adjusted = SID_ap_new * magnification_scale_factor_ap
        # --- Use hardcoded detector parameters for both AP and LP ---
        det_size_x_hard = 1052
        det_size_y_hard = 1024
        pixel_spacing_hard = 0.741
        # Remove or comment out the following lines:
        # det_size_x_lp_scaled = int(round(det_size_x_lp * fov_scale_factor))
        # det_size_y_lp_scaled = int(round(det_size_y_lp * fov_scale_factor))
        # det_size_x_ap_scaled = int(round(det_size_x_ap * fov_scale_factor))
        # det_size_y_ap_scaled = int(round(det_size_y_ap * fov_scale_factor))
        # Instead, use the hardcoded values:
        det_size_x_lp_scaled = det_size_x_hard
        det_size_y_lp_scaled = det_size_y_hard
        det_size_x_ap_scaled = det_size_x_hard
        det_size_y_ap_scaled = det_size_y_hard
        det_spacing_x_lp = pixel_spacing_hard
        det_spacing_y_lp = pixel_spacing_hard
        det_spacing_x_ap = pixel_spacing_hard
        det_spacing_y_ap = pixel_spacing_hard
        # ---
        lp_xy_angle, lp_yz_angle = estimate_angle_from_direction(direction_lp_new)
        ap_xy_angle, ap_yz_angle = estimate_angle_from_direction(direction_ap_new)
        drr_lp = generate_drr_mbircone(volume_attenuation, ct_spacing_zyx, SAD_lp_new, SID_lp_adjusted,
            det_size_x_lp_scaled, det_size_y_lp_scaled, det_spacing_x_lp, det_spacing_y_lp,
            rotation_angle_deg=lp_xy_angle)
        drr_ap = generate_drr_mbircone(volume_attenuation, ct_spacing_zyx, SAD_ap_new, SID_ap_adjusted,
            det_size_x_ap_scaled, det_size_y_ap_scaled, det_spacing_x_ap, det_spacing_y_ap,
            rotation_angle_deg=ap_xy_angle)
        plt.imshow(drr_lp, cmap='gray')
        plt.title("Raw LP DRR (before VTK display)")
        plt.show()
        plt.imshow(drr_ap, cmap='gray')
        plt.title("Raw AP DRR (before VTK display)")
        plt.show()
        vec_lp = isocenter_ct_hivtk - source_lp_new
        vec_lp /= np.linalg.norm(vec_lp)
        vec_ap = isocenter_ct_hivtk - source_ap_new
        vec_ap /= np.linalg.norm(vec_ap)
        visualize_camera_setup(source_lp_new, source_ap_new, isocenter_ct_hivtk, direction_lp=vec_lp, direction_ap=vec_ap)
        # --- Apply flips before normalization and saving ---
        # 1. LP: horizontal flip (left-right)
        drr_lp_flipped = np.fliplr(drr_lp)
        # 2. AP: no horizontal flip needed
        drr_ap_flipped = drr_ap

        # --- Normalize for display and saving (no vertical flip) ---
        drr_lp_normalized = normalize_drr_for_display(drr_lp_flipped)
        drr_ap_normalized = normalize_drr_for_display(drr_ap_flipped)

        # --- Save PNG and DICOM using normalized, flipped images (no vertical flip) ---
        # For saving: flip vertically to match matplotlib orientation
        drr_lp_to_save = np.flipud(drr_lp_normalized)
        drr_ap_to_save = np.flipud(drr_ap_normalized)

        if drr_lp is not None:
            save_vtk_image_as_png(numpy_to_vtk_image_data(drr_lp_to_save), os.path.join(output_dir, "drr_lp.png"))
            seq_num_lp = get_next_sequence_number(output_dir, "LP")
            save_drr_as_dicom(drr_lp_to_save, [float(det_spacing_y_lp), float(det_spacing_x_lp)], output_dir, f"LP_{seq_num_lp}.dcm", metadata, "LP")
        if drr_ap is not None:
            save_vtk_image_as_png(numpy_to_vtk_image_data(drr_ap_to_save), os.path.join(output_dir, "drr_ap.png"))
            seq_num_ap = get_next_sequence_number(output_dir, "AP")
            save_drr_as_dicom(drr_ap_to_save, [float(det_spacing_y_ap), float(det_spacing_x_ap)], output_dir, f"AP_{seq_num_ap}.dcm", metadata, "AP")

        # For VTK display, use the already vertically flipped arrays
        lp_window, lp_interactor, _ = create_image_viewer(drr_lp_to_save, "LP DRR", "LP DRR", output_dir, metadata,
            [det_spacing_y_lp, det_spacing_x_lp], "LP")
        ap_window, ap_interactor, _ = create_image_viewer(drr_ap_to_save, "AP DRR", "AP DRR", output_dir, metadata,
            [det_spacing_y_ap, det_spacing_x_ap], "AP")

        if lp_window: lp_window.SetPosition(700, 50)
        if ap_window: ap_window.SetPosition(700, 700)
        if lp_interactor: lp_interactor.Initialize()
        if ap_interactor: ap_interactor.Initialize()
        if lp_window: lp_window.Render()
        if ap_window: ap_window.Render()
        if lp_interactor: lp_interactor.Start()
        if ap_interactor: ap_interactor.Start()
        print("==== CT VOLUME INFO ====")
        print("CT volume shape (z, y, x):", volume_hu.shape)
        print("CT spacing (z, y, x):", ct_spacing_zyx)
        print("CT origin (x, y, z):", origin)
        print()
        print("==== PROJECTION MATRIX INFO ====")
        print("AP source position (from P2):", source_ap)
        print("LP source position (from P1):", source_lp)
        print("AP detector center (from P2):", detector_center_ap)
        print("LP detector center (from P1):", detector_center_lp)
        print()
        print("==== DIRECTION VECTORS ====")
        print("AP direction vector (from P2):", direction_ap_new)
        print("LP direction vector (from P1):", direction_lp_new)
        print()
        print("==== CALCULATED ANGLES ====")
        print("AP XY angle (degrees):", ap_xy_angle)
        print("LP XY angle (degrees):", lp_xy_angle)
        print()
        print("==== DRR GENERATION ====")
        print(f"Generating AP DRR with angle: {ap_xy_angle:.2f} degrees")
        print(f"Generating LP DRR with angle: {lp_xy_angle:.2f} degrees")
        print()
        row_idx = drr_lp.shape[0] // 2  # Middle row
        print("Middle row of raw LP DRR:", drr_lp[row_idx, :10])
        print("Middle row of flipped LP DRR:", drr_lp_flipped[row_idx, :10])

        plt.figure()
        plt.imshow(drr_lp, cmap='gray')
        plt.title('Raw LP DRR')
        plt.show()

        plt.figure()
        plt.imshow(drr_lp_flipped, cmap='gray')
        plt.title('Flipped LP DRR')
        plt.show()
    except Exception as e:
        print(f"\nError in DRR generation or visualization: {str(e)}")


if __name__ == "__main__":
    from contextlib import redirect_stderr
    class DevNull:
        def write(self, msg): pass
        def flush(self): pass
    with redirect_stderr(DevNull()):
        main()