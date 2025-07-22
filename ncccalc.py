import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSpinBox, QFormLayout, QDoubleSpinBox, QMessageBox, 
    QSlider, QSizePolicy, QGroupBox, QScrollArea, QGridLayout, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class XrayProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-ray Image Processor with Filters")
        self.resize(800, 600)
        self.setMinimumSize(600, 400)
        self.setStyleSheet(self.get_stylesheet())

        # Initialize variables
        self.original_image = None
        self.log_image = None
        self.flat = None
        self.dark = None
        self.final_image = None
        self.processed = None
        self.drr_image = None
        self.filtered_image = None
        self.filename = None
        self.filter_type = "Median"

        # Processing parameters
        self.bit_depth = 8
        self.air_val = 255.0
        self.min_val = 1e-6
        self.inlier_threshold = 100.0

        # Windowing
        self.window_center = 128
        self.window_width = 256
        self.dragging = False
        self.last_x = None
        self.last_y = None

        # Create UI
        self.init_ui()

    def init_ui(self):
        # --- Create All UI Components ---
        self.create_buttons()
        self.create_image_views()
        self.create_parameter_controls()
        self.create_window_controls()
        self.create_filter_controls()
        self.create_histogram()

        # --- Layout ---
        main_layout = QHBoxLayout()

        # Left Panel
        left_panel = QVBoxLayout()
        left_panel.addWidget(self.create_process_group())
        left_panel.addWidget(self.create_params_group())
        left_panel.addWidget(self.create_window_group())
        left_panel.addWidget(self.create_filter_group())

        # Middle Panel (image display)
        orig_layout = QVBoxLayout()
        orig_layout.addWidget(self.orig_label)
        orig_layout.addWidget(self.orig_caption)

        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_label)
        log_layout.addWidget(self.log_caption)

        final_layout = QVBoxLayout()
        final_layout.addWidget(self.final_label)
        final_layout.addWidget(self.final_caption)

        drr_layout = QVBoxLayout()
        drr_layout.addWidget(self.drr_label)
        drr_layout.addWidget(self.drr_caption)

        proc_layout = QVBoxLayout()
        proc_layout.addWidget(self.proc_label)
        proc_layout.addWidget(self.proc_caption)

        filtered_layout = QVBoxLayout()
        filtered_layout.addWidget(self.filtered_label)
        filtered_layout.addWidget(self.filtered_caption)

        image_row1 = QHBoxLayout()
        image_row1.addLayout(orig_layout)
        image_row1.addLayout(log_layout)
        image_row1.addLayout(final_layout)

        image_row2 = QHBoxLayout()
        image_row2.addLayout(drr_layout)
        image_row2.addLayout(proc_layout)
        image_row2.addLayout(filtered_layout)

        image_vlayout = QVBoxLayout()
        image_vlayout.addLayout(image_row1)
        image_vlayout.addLayout(image_row2)

        image_group = QGroupBox("Images")
        image_group.setLayout(image_vlayout)

        middle_panel = QVBoxLayout()
        middle_panel.addWidget(image_group)

        # Right Panel
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.create_processed_view_group())
        right_panel.addWidget(self.histogram_group)

        # Combine all panels
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(middle_panel, 2)
        main_layout.addLayout(right_panel, 2)

        # Scroll Area
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(main_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        # Set final layout
        layout = QVBoxLayout(self)
        layout.addWidget(scroll_area)

        self.update_button_states()

        
    def create_buttons(self):
        # Workflow buttons
        self.load_btn = QPushButton("Load X-ray Image")
        self.log_btn = QPushButton("Log Transform & Save")
        self.gen_flat_dark_btn = QPushButton("Generate Flat/Dark")
        self.process_btn = QPushButton("Apply Gain Correction")
        self.load_drr_btn = QPushButton("Load DRR (Reference)")
        self.show_hist_btn = QPushButton("Show Histogram")
        self.save_proc_btn = QPushButton("Save Current Image")
        
        # Connect signals
        self.load_btn.clicked.connect(self.load_image)
        self.log_btn.clicked.connect(self.apply_log_conversion)
        self.gen_flat_dark_btn.clicked.connect(self.generate_flat_dark)
        self.process_btn.clicked.connect(self.apply_gain)
        self.load_drr_btn.clicked.connect(self.load_drr)
        self.show_hist_btn.clicked.connect(self.toggle_histogram)
        self.save_proc_btn.clicked.connect(self.save_image)
        
    def create_filter_controls(self):
        # Filter selection
        self.filter_select = QComboBox()
        self.filter_select.addItems(["Median", "Gaussian"])
        self.filter_select.currentTextChanged.connect(self.change_filter)
        
        # Filter size slider
        self.filter_size_slider = QSlider(Qt.Horizontal)
        self.filter_size_slider.setRange(1, 21)
        self.filter_size_slider.setSingleStep(2)
        self.filter_size_slider.setPageStep(2)
        self.filter_size_slider.setValue(3)
        self.filter_size_slider.setTickInterval(2)
        self.filter_size_slider.setTickPosition(QSlider.TicksBelow)
        self.filter_size_slider.valueChanged.connect(self.apply_filter)  # Live filter effect
        
        # Apply filter button
        self.apply_filter_btn = QPushButton("Apply Filter")
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        
    def create_filter_group(self):
        group = QGroupBox("Image Filtering")
        layout = QFormLayout()
        
        layout.addRow("Filter Type:", self.filter_select)
        layout.addRow("Filter Size:", self.filter_size_slider)
        layout.addRow(self.apply_filter_btn)
        
        group.setLayout(layout)
        return group    
        
    def create_image_views(self):
        # Image labels
        self.orig_label = QLabel()
        self.orig_caption = QLabel("Original X-ray")
        self.orig_caption.setAlignment(Qt.AlignCenter)
        self.orig_caption.setStyleSheet("color: lightgray;")

        self.log_label = QLabel()
        self.log_caption = QLabel("Log Transformed")
        self.log_caption.setAlignment(Qt.AlignCenter)
        self.log_caption.setStyleSheet("color: lightgray; font-size: 12px;")

        self.final_label = QLabel()
        self.final_caption = QLabel("Gain Corrected Image")
        self.final_caption.setAlignment(Qt.AlignCenter)
        self.final_caption.setStyleSheet("color: lightgray; font-size: 12px;")

        self.drr_label = QLabel()
        self.drr_caption = QLabel("DRR Reference")
        self.drr_caption.setAlignment(Qt.AlignCenter)
        self.drr_caption.setStyleSheet("color: lightgray; font-size: 12px;")

        self.proc_label = QLabel()
        self.proc_caption = QLabel("Window Level Adjusted")
        self.proc_caption.setAlignment(Qt.AlignCenter)
        self.proc_caption.setStyleSheet("color: lightgray; font-size: 12px;")

        self.filtered_label = QLabel()
        self.filtered_caption = QLabel("Filtered Applied")
        self.filtered_caption.setAlignment(Qt.AlignCenter)
        self.filtered_caption.setStyleSheet("color: lightgray; font-size: 12px;")
        
        # Configure labels
        for label in [self.orig_label, self.log_label, self.final_label, 
                     self.drr_label, self.proc_label, self.filtered_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: #2a2a2a; color: white; padding: 5px; border-radius: 5px;")
            label.setMinimumSize(100, 100)
            label.setMaximumSize(300, 300)  # Allow resizing but not too large
            
    def create_parameter_controls(self):
        # Bit depth
        self.bit_depth_spin = QSpinBox()
        self.bit_depth_spin.setRange(8, 16)
        self.bit_depth_spin.setValue(8)
        self.bit_depth_spin.setReadOnly(True)
        
        # Min value
        self.min_val_spin = QDoubleSpinBox()
        self.min_val_spin.setDecimals(6)
        self.min_val_spin.setRange(0.0, 1.0)
        self.min_val_spin.setSingleStep(1e-6)
        self.min_val_spin.setValue(1e-6)
        
        # Inlier threshold
        self.inlier_thresh_spin = QDoubleSpinBox()
        self.inlier_thresh_spin.setRange(0.0, 500.0)
        self.inlier_thresh_spin.setSingleStep(0.1)
        self.inlier_thresh_spin.setValue(100.0)
        
           # Air value
        self.air_val_spin = QDoubleSpinBox()
        self.air_val_spin.setDecimals(6)
        self.air_val_spin.setRange(0.0, 255.0)  # adjust range as per your data
        self.air_val_spin.setSingleStep(0.01)
        self.air_val_spin.setValue(255.0)  # a typical value post-log for air
        
    def create_window_controls(self):
        # Window center slider
        self.slider_center = QSlider(Qt.Horizontal)
        self.slider_center.setRange(0, 255)
        self.slider_center.setValue(128)
        self.slider_center.valueChanged.connect(self.update_display_from_slider)
        
        # Window width slider
        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setRange(1, 255)
        self.slider_width.setValue(128)
        self.slider_width.valueChanged.connect(self.update_display_from_slider)
        
        # Value labels
        self.center_value_label = QLabel("128")
        self.width_value_label = QLabel("128")
        
    def create_histogram(self):
        self.histogram_group = QGroupBox("Histogram")
        self.histogram_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 8px; }")
        
        # Create figure
        self.figure = Figure(figsize=(5, 4), facecolor='#1a1a1a')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Configure axes
        self.ax.set_facecolor('#2a2a2a')
        self.ax.tick_params(axis='x', colors='#c0c0c0')
        self.ax.tick_params(axis='y', colors='#c0c0c0')
        self.ax.spines['bottom'].set_color('#444')
        self.ax.spines['top'].set_color('#444')
        self.ax.spines['left'].set_color('#444')
        self.ax.spines['right'].set_color('#444')
        
        # Hide initially
        self.canvas.hide()
        
        # Layout for histogram
        hist_layout = QVBoxLayout()
        hist_layout.addWidget(self.canvas)
        self.histogram_group.setLayout(hist_layout)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
    def create_process_group(self):
        group = QGroupBox("X-ray Image Processing Workflow")
        layout = QVBoxLayout()
        layout.addWidget(self.load_btn)
        layout.addWidget(self.log_btn)
        layout.addWidget(self.gen_flat_dark_btn)
        layout.addWidget(self.process_btn)
        layout.addWidget(self.load_drr_btn)
        layout.addWidget(self.show_hist_btn)
        layout.addWidget(self.save_proc_btn)
        group.setLayout(layout)
        return group
        
    def create_params_group(self):
        group = QGroupBox("Parameters")
        layout = QFormLayout()
        layout.addRow("Bit Depth", self.bit_depth_spin)
        layout.addRow("Air Value", self.air_val_spin)
        layout.addRow("Min Val (for Log)", self.min_val_spin)
        layout.addRow("Inlier Threshold", self.inlier_thresh_spin)
        group.setLayout(layout)
        return group
        
    def create_window_group(self):
        group = QGroupBox("Windowing Control")
        layout = QFormLayout()
        
        self.create_window_controls()
        
        layout.addRow("Window Center", self.slider_center)
        layout.addRow("", self.center_value_label)
        layout.addRow("Window Width", self.slider_width)
        layout.addRow("", self.width_value_label)
        
        group.setLayout(layout)
        return group
        
    def create_image_display_group(self):
        group = QGroupBox("Image Views")
        layout = QGridLayout()
        
        layout.addWidget(self.orig_label, 0, 0)
        layout.addWidget(self.log_label, 0, 1)
        layout.addWidget(self.final_label, 1, 0)
        layout.addWidget(self.drr_label, 1, 1)
        layout.addWidget(self.filtered_label, 2, 0, 1, 2)  # Span two columns
        
        group.setLayout(layout)
        return group
        
    def create_processed_view_group(self):
        group = QGroupBox("Processed Image & NCC")
        layout = QVBoxLayout()
        
        self.ncc_label = QLabel("NCC Score: N/A")
        self.ncc_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.proc_label)
        layout.addWidget(self.ncc_label)
        group.setLayout(layout)
        return group
        
    def get_stylesheet(self):
        return """
        QWidget {
            background-color: #1a1a1a;
            color: #e0e0e0;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 15px;
        }
        QPushButton {
            background-color: #007acc;
            color: white;
            padding: 10px 15px;
            border-radius: 6px;
            border: none;
            font-weight: bold;
            text-transform: uppercase;
            margin: 5px 0;
        }
        QPushButton:hover {
            background-color: #005f9a;
        }
        QPushButton:disabled {
            background-color: #333333;
            color: #888888;
            border: 1px solid #555555;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #444;
            border-radius: 8px;
            margin-top: 10px;
            padding: 10px;
            background-color: #2a2a2a;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 8px;
            color: #a0a0a0;
            font-size: 15px;
        }
        QLabel {
            border: 1px solid #333;
            border-radius: 5px;
            padding: 5px;
            color: #c0c0c0;
        }
        QSlider::groove:horizontal {
            border: 1px solid #333;
            height: 8px;
            background: #555;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #007acc;
            border: 1px solid #005f9a;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: #3a3a3a;
            color: #e0e0e0;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 3px;
        }
        QComboBox {
            background-color: #3a3a3a;
            color: #e0e0e0;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 3px;
        }
        """
        
    def update_button_states(self):
        self.load_btn.setEnabled(True)
        self.log_btn.setEnabled(self.original_image is not None)
        self.gen_flat_dark_btn.setEnabled(self.log_image is not None)
        self.process_btn.setEnabled(self.flat is not None and self.dark is not None)
        self.load_drr_btn.setEnabled(self.final_image is not None)
        self.show_hist_btn.setEnabled(self.final_image is not None)
        self.save_proc_btn.setEnabled(self.processed is not None)
        self.apply_filter_btn.setEnabled(self.final_image is not None)  # Enable filter button if processed image is available
        
        self.slider_center.setEnabled(self.final_image is not None)
        self.slider_width.setEnabled(self.final_image is not None)
        
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open X-ray Image", "", 
            "Images (*.png *.jpg *.bmp *.tif *.dcm);;All Files (*)"
        )
        
        if path:
            self.original_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.filename = os.path.basename(path)
            
            if self.original_image is not None:
                if self.original_image.dtype == np.uint8:
                    self.bit_depth = 8
                    self.air_val = 255.0
                elif self.original_image.dtype == np.uint16:
                    self.bit_depth = 16
                    self.air_val = 65535.0
                else:
                    QMessageBox.warning(self, "Unsupported Format", 
                                      "Unsupported bit depth or image format. Please use 8-bit or 16-bit images.")
                    self.original_image = None
                    self.update_button_states()
                    return
                
                self.bit_depth_spin.setValue(self.bit_depth)
                self.show_image(self.orig_label, self.original_image)
                
                # Reset other states
                self.log_image = None
                self.final_image = None
                self.processed = None
                self.drr_image = None
                self.filtered_image = None
                self.update_button_states()
                
                # Apply default filter
                self.apply_filter()
            else:
                QMessageBox.critical(self, "Load Error", 
                                   "Could not load the image. Please ensure it's a valid image file.")
                
    def apply_log_conversion(self):
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Load an X-ray image first.")
            return
            
        try:
            img_float = self.original_image.astype(np.float32) + 1.0
            self.log_image = np.log(img_float)
            
            # Normalize for display
           
            log_img_norm = cv2.normalize(self.log_image, None, 0, 255, cv2.NORM_MINMAX)
            self.log_image = log_img_norm.astype(np.uint8)
            
            self.show_image(self.log_label, self.log_image)
            self.update_button_states()
        except Exception as e:
            QMessageBox.critical(self, "Log Transformation Error", 
                               f"Failed to apply log transformation: {e}")
            
    def generate_flat_dark(self):
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Load an X-ray image first.")
            return
            
        try:
            max_val = 2**self.bit_depth - 1
            h, w = self.original_image.shape[:2]
            
            # Create flat field with gradient and noise
            x_grad = np.linspace(0.9, 1.1, w)
            y_grad = np.linspace(0.95, 1.05, h)
            flat = np.ones((h, w), dtype=np.float32) * (max_val * 0.94)
            flat *= np.outer(y_grad, x_grad)
            flat += np.random.normal(0, max_val * 0.01, (h, w))
            
            # Create dark field with noise
            dark = np.random.normal(max_val * 0.02, max_val * 0.005, (h, w))
            
            # Clip and convert
            flat = np.clip(flat, 0, max_val).astype(np.uint16 if self.bit_depth == 16 else np.uint8)
            dark = np.clip(dark, 0, max_val).astype(np.uint16 if self.bit_depth == 16 else np.uint8)
            
            # Store as float32 for calculations
            self.flat = flat.astype(np.float32)
            self.dark = dark.astype(np.float32)
            
            QMessageBox.information(self, "Success", 
                                  "Simulated Flat and Dark images generated successfully!")
            self.update_button_states()
        except Exception as e:
            QMessageBox.critical(self, "Flat/Dark Generation Error", 
                               f"Failed to generate flat/dark images: {e}")
            
    def apply_gain(self):
        if self.original_image is None or self.flat is None or self.dark is None:
            QMessageBox.warning(self, "Missing Inputs", 
                              "Please ensure Original image, Flat image, and Dark image are available.")
            return
            
        try:
            self.min_val = self.min_val_spin.value()
            self.inlier_threshold = self.inlier_thresh_spin.value()
            
            img = self.original_image.astype(np.float32)
            flat_dark = self.flat - self.dark
            denom = flat_dark.copy()
            denom[denom <= 0] = self.min_val
            
            corrected = (img - self.dark) / denom
            corrected = np.clip(corrected, self.min_val, None)
            
            # Apply negative logarithm
            log_img = -np.log(corrected)
            log_img = np.max(log_img) - log_img
            # Inlier masking
            low_perc = self.inlier_threshold
            high_perc = 100 - self.inlier_threshold
            low_perc = np.clip(low_perc, 0, 100)
            high_perc = np.clip(high_perc, 0, 100)
            
            if low_perc > high_perc:
                low_perc, high_perc = high_perc, low_perc
                
            flat_dark_pixels = flat_dark.flatten()
            if len(flat_dark_pixels) > 0:
                low_threshold, high_threshold = np.percentile(flat_dark_pixels, [low_perc, high_perc])
            else:
                low_threshold, high_threshold = 0, 0
                
            mask = (flat_dark >= low_threshold) & (flat_dark <= high_threshold)
            
            if np.sum(mask) == 0:
                QMessageBox.warning(self, "Inliers Failed", 
                                  "No valid pixels found for inlier masking. Adjust Inlier Threshold.")
            else:
                median_val = np.median(log_img[mask])
                log_img[~mask] = median_val
                
            # Normalize for display
            min_log = np.min(log_img)
            max_log = np.max(log_img)
            
            if np.isclose(max_log, min_log, atol=1e-8):
                norm = np.zeros_like(log_img)
            else:
                norm = (log_img - min_log) / (max_log - min_log + 1e-8)
                
            norm_uint8 = (norm * 255).astype(np.uint8)
            
            self.final_image = norm_uint8
            self.processed = norm_uint8.copy()  # Store the processed image
            self.show_image(self.final_label, self.final_image)
            
            self.update_display()
            self.update_button_states()
            QMessageBox.information(self, "Success", "Gain correction applied successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Gain Correction Error", 
                               f"Failed to apply gain correction: {e}")
            print(f"Error in apply_gain: {e}")
            
    def load_drr(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load DRR (Reference)", "", 
            "Images (*.png *.jpg *.bmp *.tif);;All Files (*)"
        )
        
        if path:
            self.drr_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.drr_image is None:
                QMessageBox.critical(self, "Load Error", 
                                   "Could not load the DRR image. Please ensure it's a valid image file.")
                return
                
            if self.processed is not None and self.drr_image.shape != self.processed.shape:
                self.drr_image = cv2.resize(self.drr_image, 
                                          (self.processed.shape[1], self.processed.shape[0]), 
                                          interpolation=cv2.INTER_AREA)
                
            self.show_image(self.drr_label, self.drr_image)
            self.update_ncc()
            self.update_button_states()

    def apply_filter(self):
        """Apply median/gaussian filter to the processed image (histogram output)"""
        if self.processed is None:
            return

        try:
            ksize = self.filter_size_slider.value()
            if ksize % 2 == 0:  # Ensure odd kernel size
                ksize += 1
                self.filter_size_slider.setValue(ksize)

            if self.filter_type == "Median":
                self.filtered_image = cv2.medianBlur(self.processed, ksize)
            elif self.filter_type == "Gaussian":
                self.filtered_image = cv2.GaussianBlur(self.processed, (ksize, ksize), 0)

            self.show_image(self.filtered_label, self.filtered_image)
            self.update_button_states()
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", f"Failed to apply {self.filter_type} filter: {e}")
            
    def change_filter(self, text):
        self.filter_type = text
        self.apply_filter()  # Re-apply filter immediately when type changes

    def apply_window_leveling(self, img):
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            
        center = self.window_center
        width = self.window_width
        
        min_val = center - width / 2
        max_val = center + width / 2
        
        img_clipped = np.clip(img, min_val, max_val)
        
        denominator = max_val - min_val
        if np.isclose(denominator, 0.0, atol=1e-6):
            img_norm = np.zeros_like(img_clipped)
        else:
            img_norm = (img_clipped - min_val) / (denominator + 1e-6)
            
        img_uint8 = (img_norm * 255).astype(np.uint8)
        return img_uint8
        
    def update_display_from_slider(self):
        self.window_center = self.slider_center.value()
        self.window_width = self.slider_width.value()
        self.center_value_label.setText(str(self.window_center))
        self.width_value_label.setText(str(self.window_width))
        self.update_display()
        
    def update_display(self):
        if self.final_image is None:
            self.show_image(self.proc_label, None)
            self.ncc_label.setText("NCC Score: N/A")
            return
            
        self.processed = self.apply_window_leveling(self.final_image)
        self.show_image(self.proc_label, self.processed)
        
        if self.canvas.isVisible():
            self.update_histogram()
            
        self.update_ncc()
        self.apply_filter()  # Apply filter whenever display updates
        
    def update_histogram(self):
        if not hasattr(self, 'final_image') or self.final_image is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Load/Process image to see histogram",
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, color='gray', fontsize=12)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            return
            
        self.ax.clear()
        flat_img = self.final_image.flatten()
        
        # Remove any existing lines
        for line in self.ax.lines:
            line.remove()
            
        # Calculate histogram data manually
        hist, bins = np.histogram(flat_img, bins=256, range=(0, 255))
        
        # Plot the histogram bars
        self.ax.bar(bins[:-1], hist, width=(bins[1]-bins[0]), color='gray', alpha=0.7)
        
        # Calculate window limits
        min_val = max(0, self.window_center - self.window_width / 2)
        max_val = min(255, self.window_center + self.window_width / 2)
        
        # Create vertical lines for windowing
        if min_val < 255:  # Only draw if it's in range
            self.ax.axvline(x=min_val, color='lime', linestyle='--', linewidth=2, label='Window Min')
        if max_val > 0:    # Only draw if it's in range
            self.ax.axvline(x=max_val, color='tomato', linestyle='--', linewidth=2, label='Window Max')
        
        self.ax.set_title("Histogram with Windowing (Drag to Adjust)", color='#e0e0e0')
        self.ax.set_xlabel("Pixel Intensity", color='#c0c0c0')
        self.ax.set_ylabel("Count", color='#c0c0c0')
        self.ax.set_xlim(0, 255)
        
        # Set background colors
        self.ax.set_facecolor('#2a2a2a')
        self.figure.set_facecolor('#1a1a1a')
        
        if len(self.ax.lines) > 0:
            self.ax.legend(facecolor='#3a3a3a', edgecolor='#555', labelcolor='white')
            
        self.canvas.draw()
        
    def update_ncc(self):
        if self.drr_image is None or self.processed is None:
            self.ncc_label.setText("NCC Score: N/A")
            return
            
        try:
            # Ensure same size
            if self.drr_image.shape != self.processed.shape:
                drr_img = cv2.resize(self.drr_image, 
                                   (self.processed.shape[1], self.processed.shape[0]), 
                                   interpolation=cv2.INTER_AREA)
            else:
                drr_img = self.drr_image
                
            # Convert to float and normalize
            img1 = self.processed.astype(np.float32)
            img2 = drr_img.astype(np.float32)
            
            # Mean subtraction
            img1 -= np.mean(img1)
            img2 -= np.mean(img2)
            
            # Standard deviation normalization
            img1 /= (np.std(img1) + 1e-8)
            img2 /= (np.std(img2) + 1e-8)
            
            # Compute correlation
            ncc = np.sum(img1 * img2) / (img1.size)
            
            self.ncc_label.setText(f"NCC Score: {ncc:.4f}")
        except Exception as e:
            self.ncc_label.setText("NCC Score: Error")
            print(f"[ERROR] NCC calculation failed: {e}")
            
    def toggle_histogram(self):
        if self.canvas.isVisible():
            self.canvas.hide()
        else:
            if self.final_image is not None:
                self.update_histogram()
                self.canvas.show()
            else:
                QMessageBox.warning(self, "No Image", 
                                  "Please process an image first to show the histogram.")
                
    def save_image(self):
        if self.filtered_image is None:
            QMessageBox.warning(self, "No Image", 
                              "No filtered image to save. Apply a filter first.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"filtered_xray_{timestamp}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Filtered Image", default_name, 
            "PNG Images (*.png);;TIFF Images (*.tif);;All Files (*)"
        )
        
        if path:
            try:
                cv2.imwrite(path, self.filtered_image)
                QMessageBox.information(self, "Saved", f"Filtered image saved to: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save image: {e}")
                
    def show_image(self, label, img):
        if img is None:
            label.clear()
            label.setText("No Image Loaded")
            return
            
        # Convert to 8-bit if needed
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
        # Convert to QImage
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888)
            
        # Convert to QPixmap and scale
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        self.dragging = True
        self.last_x = event.xdata
        self.last_y = event.ydata
        
    def on_drag(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
            
        if self.last_x is None or self.last_y is None:
            return
            
        dx = event.xdata - self.last_x if event.xdata is not None else 0
        dy = event.ydata - self.last_y if event.ydata is not None else 0
        
        # Update windowing parameters
        self.window_center = int(np.clip(self.window_center + dx, 0, 255))
        self.window_width = int(np.clip(self.window_width - dy, 1, 255))
        
        # Update sliders and display
        self.slider_center.setValue(self.window_center)
        self.slider_width.setValue(self.window_width)
        self.center_value_label.setText(str(self.window_center))
        self.width_value_label.setText(str(self.window_width))
        
        self.last_x = event.xdata
        self.last_y = event.ydata
        
        self.update_display()
        
    def on_release(self, event):
        self.dragging = False
        self.last_x = None
        self.last_y = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = XrayProcessor()
    win.show()
    sys.exit(app.exec_())
