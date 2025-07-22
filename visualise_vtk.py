
import vtk
import numpy as np
import pydicom
import json
from scipy.spatial.transform import Rotation as R
import time 
import os

def create_ct_homogeneous_transformation(origin, spacing, direction):
    """
    Create a 4x4 homogeneous transformation matrix using VTK based on the origin, spacing, and direction.
    """
    if len(direction) == 6:
        direction = list(direction) + [0.0, 0.0, 1.0]
    direction_matrix = np.array(direction).reshape(3, 3)
    print(f"Direction Matrix {direction_matrix}")
    transform = vtk.vtkTransform()
    transform.Translate(origin[0], origin[1], origin[2])
    transform.Scale(spacing[0], spacing[1], spacing[2])
    rotation_matrix = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            rotation_matrix.SetElement(i, j, direction_matrix[i, j])
    transform.Concatenate(rotation_matrix)
    # Extract the resulting 4x4 transformation matrix
    vtk_matrix = vtk.vtkMatrix4x4()
    transform.GetMatrix(vtk_matrix)

    # Convert to a numpy array for easier handling
    transformation_matrix_4x4 = np.array([[vtk_matrix.GetElement(i, j) for j in range(4)] for i in range(4)])

    return transformation_matrix_4x4


def drawlinesfromorigin():
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)  # Origin
    points.InsertNextPoint(500, 0, 0)  # X-axis point
    points.InsertNextPoint(0, 500, 0)  # Y-axis point
    points.InsertNextPoint(0, 0, 500)  # Z-axis point

    # Create lines for each axis
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(2)  # Line for X-axis
    lines.InsertCellPoint(0)  # Start point (origin)
    lines.InsertCellPoint(1)  # End point (500, 0, 0)

    lines.InsertNextCell(2)  # Line for Y-axis
    lines.InsertCellPoint(0)  # Start point (origin)
    lines.InsertCellPoint(2)  # End point (0, 500, 0)

    lines.InsertNextCell(2)  # Line for Z-axis
    lines.InsertCellPoint(0)  # Start point (origin)
    lines.InsertCellPoint(3)  # End point (0, 0, 500)

    # Create a polydata to hold the points and lines
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    # Create an actor for each line with different colors
    x_axis_actor = vtk.vtkActor()
    x_axis_actor.SetMapper(mapper)
    x_axis_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red color for X-axis

    y_axis_actor = vtk.vtkActor()
    y_axis_actor.SetMapper(mapper)
    y_axis_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green color for Y-axis

    z_axis_actor = vtk.vtkActor()
    z_axis_actor.SetMapper(mapper)
    z_axis_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # Blue color for Z-axis
    
    return x_axis_actor,y_axis_actor,z_axis_actor

def getsphere(ip):
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetCenter(ip[0],ip[1],ip[2])  # Set the position to the origin
    sphere_source.SetRadius(5.0)
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())
    
    # Create an actor to represent the sphere
    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    
    return sphere_actor

def display_ct_volume(reader):
    """
    Display the CT volume and allow picking of points.
    Returns the render window and interactor for external control.
    """
    # Create 3D volume actor
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputConnection(reader.GetOutputPort())

    volumeColor = vtk.vtkColorTransferFunction()
    # Background or low intensities
    volumeColor.AddRGBPoint(0, 0.0, 0.0, 0.0)       # Black (background)
    volumeColor.AddRGBPoint(100, 0.5, 0.2, 0.1)     # Dark reddish-brown

    # Transition region
    volumeColor.AddRGBPoint(500, 1.0, 0.8, 0.6)     # Light reddish-brown

    # Bone region
    volumeColor.AddRGBPoint(1000, 1.0, 1.0, 1.0)    # White
    volumeColor.AddRGBPoint(1150, 1.0, 1.0, 1.0) 

    volumeScalarOpacity = vtk.vtkPiecewiseFunction()
    volumeScalarOpacity.AddPoint(0, 0.0)
    volumeScalarOpacity.AddPoint(100, 0.5)
    volumeScalarOpacity.AddPoint(500, 0.85)
    volumeScalarOpacity.AddPoint(700, 0.85)
    volumeScalarOpacity.AddPoint(1000, 0.85)
    volumeScalarOpacity.AddPoint(1150, 0.85)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(volumeColor)
    volumeProperty.SetScalarOpacity(volumeScalarOpacity)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()
    
    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(volumeMapper)
    volume_actor.SetProperty(volumeProperty)

    # Step 2: Define rotation and translation
    angle_x = -11.09  # degrees
    angle_y = -8.21
    angle_z = 166.92
    translate_x = 192.07  # mm
    translate_y = 154.24
    translate_z = 291.91
    
    # Step 3: Create vtkTransform and apply transformations
    transform = vtk.vtkTransform()
    transform.PostMultiply()  # Apply in order: rotate first, then translate
    
    # Apply rotations in desired order (e.g., XYZ)
    transform.RotateX(angle_x)
    transform.RotateY(angle_y)
    transform.RotateZ(angle_z)
    
    # Then apply translation
    transform.Translate(translate_x, translate_y, translate_z)
    
    # Step 4: Set the transform to volume actor
    volume_actor.SetUserTransform(transform)

    # Create a sphere at the origin
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetCenter(0.0, 0.0, 0.0)  # Set the position to the origin
    sphere_source.SetRadius(2.0)  # Set the radius of the sphere
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())
    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)

    # Add coordinate axes
    a, b, c = drawlinesfromorigin()
    actors_sph = []
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume_actor)
    renderer.AddActor(sphere_actor)  # Add the sphere to the renderer
    a.GetProperty().SetColor(1.0, 0.0, 0.0)
    renderer.AddActor(a)
    renderer.AddActor(b)
    renderer.AddActor(c)
    for actor_sphere in actors_sph:
        renderer.AddActor(actor_sphere)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background

    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1920, 1080)

    # Create interactor
    render_interactor = vtk.vtkRenderWindowInteractor()
    render_interactor.SetRenderWindow(render_window)

    # Custom interactor style to toggle picking
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    render_interactor.SetInteractorStyle(interactor_style)
    
    # Render once but don't start the interactor
    render_window.Render()
    
    # Return the window and interactor for external control
    return render_window, render_interactor




    
if __name__ == "__main__":
    dicom_folder = os.path.expanduser("~/Downloads/A_CT_SCAN_FOLDER")
    
    # Read the DICOM series
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dicom_folder)
    reader.Update()
    
    # Display the volume
    window, interactor = display_ct_volume(reader)
    
    # Start the interactor if running standalone
    interactor.Start()