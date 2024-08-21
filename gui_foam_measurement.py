import cv2
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
from picamera2 import Picamera2
import paho.mqtt.client as mqtt
from paho.mqtt.client import ConnectFlags, ReasonCodes
import time
import json

class FoamMeasurementApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
        self.picam2.start()
        self.picam2.set_controls({"AeEnable":False})
        time.sleep(1)  # Give the camera time to warm up
        
        self.image_lock = threading.Lock()
        self.current_image = None
        self.running = True
        self.processing_thread = threading.Thread(target=self.image_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # MQTT setup
        self.mqtt_client = mqtt.Client(protocol=mqtt.MQTTv311)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.connect("localhost", 1884, 60)
        self.mqtt_client.loop_start()
        
        self.image = None
        self.processed_image = None
        self.scale = 1.0  # pixels per ml
        self.show_edges = tk.BooleanVar(value=False)
        self.calibrating = False
        self.roi = [0, 0, 640, 480]  # [x, y, w, h]
        self.drawing_roi = False
        self.upper_edge = None
        self.lower_edge = None
        self.tuning_mode = False
        self.calibrating = False
        self.calibration_start = None
        self.calibration_end = None
        
        # table to record the result
        self.measurement_history = []
        self.history_table = None
        # Create GUI elements
        self.create_widgets()
        self.load_config()
        
        # Start video stream, schedule next update with 200ms interval
        self.window.after(200, self.update_video)
        # Set up the protocol for window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    
    def on_mqtt_connect(self, client, userdata, flags, rc, properties=None):
        rc_int = rc if isinstance(rc, int) else rc.value
        result_meanings = {
            0: "Connection successful",
            1: "Connection refused - incorrect protocol version",
            2: "Connection refused - invalid client identifier",
            3: "Connection refused - server unavailable",
            4: "Connection refused - bad username or password",
            5: "Connection refused - not authorized"
        }
        
        if rc_int in result_meanings:
            result_message = result_meanings[rc_int]
        else:
            result_message = f"Unknown result code: {rc_int}"
        
        print(f"MQTT Connection result: {result_message}")
        
        if rc_int == 0:
            print("Successfully connected to MQTT broker")
            self.mqtt_client.subscribe("trigger_cam")
        else:
            print("Failed to connect to MQTT broker. Please check your connection settings.")

    def on_mqtt_message(self, client, userdata, msg):
        if msg.topic == "trigger_cam":
            print("Received trigger from MQTT")
            self.window.after(0, self.measure_foam)

    def cleanup(self):
        print("Stopping MQTT client...")
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        
        print("Stopping image processing...")
        self.running = False
        self.processing_thread.join()  # Wait for the thread to finish
        
        print("Stopping camera...")
        self.picam2.stop()
        
        print("Cleanup complete.")
        
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            print("Closing application...")
            self.cleanup()
            self.window.destroy()        
        
    def image_processing_loop(self):
        while self.running:
            with self.image_lock:
                self.current_image = self.picam2.capture_array()
            time.sleep(0.1)  # Capture at 10 FPS
            
    def create_widgets(self):
         # Main frame
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(padx=10, pady=10)

        # Video frame
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)

        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.draw_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)

        # Control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10)

        # ROI label
        self.roi_label = ttk.Label(self.control_frame, text="ROI: Click and drag on image")
        self.roi_label.pack(pady=5)

        # Scale entry
        scale_frame = ttk.Frame(self.control_frame)
        scale_frame.pack(pady=5)
        self.scale_label = ttk.Label(scale_frame, text="Scale (px/ml):")
        self.scale_label.pack(side=tk.LEFT)
        self.scale_entry = ttk.Entry(scale_frame, width=10)
        self.scale_entry.insert(0, "1.0")
        self.scale_entry.pack(side=tk.LEFT)

        # Calibration section
        self.calibration_label = ttk.Label(scale_frame, text="Calibrate:")
        self.calibration_label.pack(side=tk.LEFT, padx=10)
        self.calibrate_button = ttk.Button(scale_frame, text="Start Calibration", command=self.start_calibration)
        self.calibrate_button.pack(side=tk.LEFT)

        self.calibration_info_label = ttk.Label(self.control_frame, text="")
        self.calibration_info_label.pack(pady=5)
        
        # Measure button
        self.measure_button = ttk.Button(self.control_frame, text="Measure Foam", command=self.measure_foam)
        self.measure_button.pack(pady=5)

        # Result label
        self.result_label = ttk.Label(self.control_frame, text="")
        self.result_label.pack(pady=5)
        # History table (initially empty)
        self.history_table = None
        self.update_history_table()

        # Tuning button
        self.tuning_button = ttk.Button(self.control_frame, text="Enter Tuning Mode", command=self.toggle_tuning_mode)
        self.tuning_button.pack(pady=5)

        # Tuning frame (initially hidden)
        self.tuning_frame = ttk.Frame(self.main_frame)
       
        # Update the threshold slider label
        self.threshold_label = ttk.Label(self.tuning_frame, text="Edge Detection Sensitivity:")
        self.threshold_label.pack()
        self.threshold_slider = ttk.Scale(self.tuning_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200, command=self.on_threshold_change)
        self.threshold_slider.set(50)  # Set a default value appropriate for Canny edge detection
        self.threshold_slider.pack()
            
        # Kernel size slider
        self.kernel_label = ttk.Label(self.tuning_frame, text="Gaussian Blur Kernel Size:")
        self.kernel_label.pack()
        self.kernel_slider = ttk.Scale(self.tuning_frame, from_=1, to=21, orient=tk.HORIZONTAL, length=200, command=self.on_kernel_change)
        self.kernel_slider.set(5)  # Set default value to 5
        self.kernel_slider.pack()
        
        self.show_edges_check = ttk.Checkbutton(self.control_frame, text="Show Live Measurment", variable=self.show_edges, command=self.on_show_edges_change)
        self.show_edges_check.state(['!selected'])
            
        self.show_edges_check.pack(pady=5)
        # Apply button
        self.apply_button = ttk.Button(self.tuning_frame, text="Apply", command=self.apply_tuning)
        self.apply_button.pack(pady=5)
        
        #Save Config button
        self.save_config_button = ttk.Button(self.control_frame, text="Save Configuration", command=self.save_config)
        self.save_config_button.pack(pady=5)
    
    def bind_roi_events(self):
        self.canvas.bind("<ButtonPress-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.draw_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)
        
    def save_config(self):
        config = {
            "scale": self.scale,
            "threshold": self.threshold_slider.get(),
            "kernel_size": self.kernel_slider.get(),
            "roi": self.roi
        }
        with open("foam_config.json", "w") as f:
            json.dump(config, f)

    def load_config(self):
        try:
            with open("foam_config.json", "r") as f:
                config = json.load(f)
            self.scale = config.get("scale", 1.0)
            self.scale_entry.delete(0, tk.END)
            self.scale_entry.insert(0, f"{self.scale:.4f}")
            self.threshold_slider.set(config.get("threshold", 128))
            self.kernel_slider.set(config.get("kernel_size", 5))
            self.roi = config.get("roi", [0, 0, 640, 480])
            self.roi_label.config(text=f"ROI: {self.roi}")
        except FileNotFoundError:
            # If the config file doesn't exist, use default values
            pass
        
    def start_calibration(self):
        self.calibrating = True
        self.calibration_start = None
        self.calibration_end = None
        self.calibrate_button.config(state="disabled")
        self.calibration_info_label.config(text="Click two points to draw a calibration line.")
        self.canvas.bind("<ButtonPress-1>", self.calibration_start_point)

    def calibration_start_point(self, event):
        self.calibration_start = (event.x, event.y)
        self.canvas.bind("<ButtonRelease-1>", self.calibration_end_point)
        self.calibration_info_label.config(text="Click the second point to complete the line.")

    def calibration_end_point(self, event):
        self.calibration_end = (event.x, event.y)
        self.draw_calibration_line()
        self.calculate_scale()
        self.calibrating = False
        self.calibrate_button.config(state="normal")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.bind_roi_events()  # Rebind ROI events after calibration

    def draw_calibration_line(self):
        if self.calibration_start and self.calibration_end:
            self.canvas.delete("calibration_line")
            self.canvas.create_line(
                self.calibration_start[0], self.calibration_start[1],
                self.calibration_end[0], self.calibration_end[1],
                fill="yellow", width=2, tags="calibration_line"
            )
            self.calibration_info_label.config(text="Calibration line drawn. Enter the known distance.")
            self.window.after(3000, self.remove_calibration_line)
        else:
            self.calibration_info_label.config(text="Please click two points to draw the calibration line.")
                
    def calculate_scale(self):
        if self.calibration_start and self.calibration_end:
            pixel_distance = ((self.calibration_end[0] - self.calibration_start[0])**2 + 
                              (self.calibration_end[1] - self.calibration_start[1])**2)**0.5
            
            known_distance = simpledialog.askfloat("Input", "Enter the known distance in ml:", 
                                               parent=self.window)
            
            if known_distance is not None and known_distance > 0:
                self.scale = pixel_distance / known_distance
                self.scale_entry.delete(0, tk.END)
                self.scale_entry.insert(0, f"{self.scale:.4f}")
                self.calibration_info_label.config(text=f"Scale set to {self.scale:.4f} pixels/ml")
            else:
                self.calibration_info_label.config(text="Invalid or cancelled distance input.")
        else:
            self.calibration_info_label.config(text="No calibration line drawn.")
   
    def remove_calibration_line(self):
        self.canvas.delete("calibration_line")
        
    def start_roi(self, event):
        self.roi = [event.x, event.y, 0, 0]
        self.drawing_roi = True

    def draw_roi(self, event):
        if self.drawing_roi:
            self.roi[2] = event.x - self.roi[0]
            self.roi[3] = event.y - self.roi[1]

    def end_roi(self, event):
        if not self.calibrating:
            self.drawing_roi = False
            self.roi[2] = max(event.x - self.roi[0], 1)
            self.roi[3] = max(event.y - self.roi[1], 1)
            self.roi_label.config(text=f"ROI: {self.roi}")
       
    def toggle_tuning_mode(self):
        self.tuning_mode = not self.tuning_mode
        if self.tuning_mode:
            self.tuning_button.config(text="Exit Tuning Mode")
            self.tuning_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.process_image()
        else:
            self.tuning_button.config(text="Enter Tuning Mode")
            self.tuning_frame.grid_forget()
            self.bind_roi_events()  # Rebind ROI events when exiting tuning mode

    def apply_tuning(self):
        self.process_image()

    def update_video(self):
        try:
            with self.image_lock:
                if self.current_image is not None:
                    self.image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

            if self.image is not None:
                display_image = self.image.copy()

                if self.tuning_mode:
                    self.process_image()
                    if self.processed_image is not None:
                        # Ensure the processed image is the same size as the ROI
                        x, y, w, h = self.roi
                        if self.processed_image.shape[:2] != (h, w):
                            self.processed_image = cv2.resize(self.processed_image, (w, h))
                        
                        # Create an overlay of the same size as the display image
                        overlay = np.zeros_like(display_image)
                        if len(self.processed_image.shape) == 2:  # If it's a grayscale image
                            overlay[y:y+h, x:x+w] = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
                        else:
                            overlay[y:y+h, x:x+w] = self.processed_image
                        
                        # Blend the overlay with the display image
                        cv2.addWeighted(display_image, 0.3, overlay, 0.7, 0, display_image)

                if self.show_edges.get():
                    self.process_image()
                    if self.upper_edge is not None and self.lower_edge is not None:
                        self.draw_irregular_edges(display_image, self.upper_edge, self.lower_edge)

                # Always draw the ROI rectangle
                x, y, w, h = self.roi
                cv2.rectangle(display_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        except Exception as e:
            print(f"Error updating video: {e}")

        # Schedule the next update
        self.window.after(100, self.update_video)
            
    def on_threshold_change(self, value):
        if self.show_edges.get():
            self.process_image()
        
    def on_kernel_change(self, value):
        if self.show_edges.get():
            self.process_image()
            
    def on_show_edges_change(self):
        if self.show_edges.get():
            self.process_image()
            
    def find_foam_edges(self, edge_image):
        h, w = edge_image.shape
        upper_edge = []
        lower_edge = None  # This will be a single y-value

        # Define a step size for horizontal scanning (e.g., every 5 or 10 pixels)
        step = 10

        # Find the lower edge first (liquid-foam interface)
        for y in range(h - 1, -1, -1):
            if np.any(edge_image[y, :]):
                lower_edge = y
                break

        if lower_edge is None:
            return None, None  # No foam detected

        # Now find the upper edge
        for x in range(0, w, step):
            col = edge_image[:lower_edge, x]  # Only search above the lower edge
            
            # Find upper edge (first non-zero pixel from top)
            upper = next((i for i, v in enumerate(col) if v != 0), None)
            if upper is not None:
                upper_edge.append((x, upper))

        # If no upper edge points were found, return None for both
        if not upper_edge:
            return None, None

        return upper_edge, lower_edge   

    
    def process_image(self):
        if self.image is None:
            return

        try:
            x, y, w, h = self.roi
            roi_image = self.image[y:y+h, x:x+w]

            gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)

            kernel_size = int(self.kernel_slider.get())
            if kernel_size % 2 == 0:
                kernel_size += 1

            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

            low_threshold = int(self.threshold_slider.get())
            high_threshold = low_threshold * 2
            edges = cv2.Canny(blurred, low_threshold, high_threshold)

            self.processed_image = edges  # This is now a binary image of the ROI only
            self.upper_edge, self.lower_edge = self.find_foam_edges(edges)

        except Exception as e:
            print(f"Error processing image: {e}")
            self.processed_image = None
            self.upper_edge, self.lower_edge = None, None

    def draw_irregular_edges(self, display_image, upper_edge, lower_edge):
        if display_image is None:
            return

        x, y, w, h = self.roi

        # Draw upper edge points
        for point in upper_edge:
            cv2.circle(display_image, (x + point[0], y + point[1]), 2, (0, 255, 0), -1)

        # Draw lower edge as a horizontal line
        cv2.line(display_image, (x, y + lower_edge), (x + w, y + lower_edge), (0, 0, 255), 2)

        # Optionally, draw a line connecting the upper edge points
        if len(upper_edge) > 1:
            upper_points = [(x + point[0], y + point[1]) for point in upper_edge]
            cv2.polylines(display_image, [np.array(upper_points)], False, (0, 255, 0), 1)

        # Draw vertical lines to show thickness at each measurement point
        for point in upper_edge:
            cv2.line(display_image, (x + point[0], y + point[1]), (x + point[0], y + lower_edge), (255, 0, 0), 1)
#     def draw_irregular_edges(self, display_image, upper_edge, lower_edge):
#         if display_image is None:
#             return
# 
#         x, y, w, h = self.roi
# 
#         # Draw upper edge points
#         for point in upper_edge:
#             cv2.circle(display_image, (x + point[0], y + point[1]), 2, (0, 255, 0), -1)
# 
#         # Draw lower edge
#         cv2.line(display_image, (x, y + lower_edge), (x + w, y + lower_edge), (0, 255, 0), 2)
        
        
    def update_history_table(self):
        if self.history_table:
            self.history_table.destroy()
        
        self.history_table = ttk.Treeview(self.control_frame, columns=('Time', 'Thickness'), show='headings', height=3)
        self.history_table.heading('Time', text='Time')
        self.history_table.heading('Thickness', text='Thickness (ml)')
        self.history_table.column('Time', width=150)
        self.history_table.column('Thickness', width=100)
        self.history_table.pack(pady=10)

        for timestamp, thickness in reversed(self.measurement_history):
            self.history_table.insert('', 'end', values=(timestamp, f"{thickness}"))
            
    def update_measurement_history(self, foam_height):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.measurement_history.append((timestamp, foam_height))
            if len(self.measurement_history) > 3:
                self.measurement_history.pop(0)
            self.update_history_table()
    def measure_foam(self):
        self.process_image()

        if self.processed_image is None:
            self.result_label.config(text="No foam detected")
            return

        try:
            self.scale = float(self.scale_entry.get())
        except ValueError:
            self.result_label.config(text="Invalid scale value")
            return

        upper_edge, lower_edge = self.find_foam_edges(self.processed_image)
        
        if upper_edge is None or lower_edge is None:
            self.result_label.config(text="No valid foam edges detected")
            return

        # Calculate average thickness
        thicknesses = [lower_edge - upper[1] for upper in upper_edge]
        if not thicknesses:
            self.result_label.config(text="No valid foam thickness measurements")
            return

        avg_thickness_px = sum(thicknesses) / len(thicknesses)
        
        foam_height = avg_thickness_px / self.scale
        result_text = f"Avg. Foam thickness: {round(foam_height)} ml"
        self.result_label.config(text=result_text)

        # Update measurement history
        self.update_measurement_history(round(foam_height))

        # Visualization
        self.draw_irregular_edges(self.image.copy(), upper_edge, lower_edge)
#     def measure_foam(self):
#         self.process_image()
# 
#         if self.processed_image is None:
#             self.result_label.config(text="No foam detected")
#             return
# 
#         try:
#             self.scale = float(self.scale_entry.get())
#         except ValueError:
#             self.result_label.config(text="Invalid scale value")
#             return
# 
#         upper_edge, lower_edge = self.find_foam_edges(self.processed_image)
#         
#         if upper_edge is None or lower_edge is None:
#             self.result_label.config(text="No foam detected")
#             return
# 
#         # Calculate average thickness
#         thicknesses = [lower_edge - upper[1] for upper in upper_edge]
#         avg_thickness_px = sum(thicknesses) / len(thicknesses)
#         
#         foam_height = avg_thickness_px / self.scale
#         result_text = f"Avg. Foam thickness: {round(foam_height)} ml"
#         self.result_label.config(text=result_text)
# 
#         # Update measurement history
#         self.update_measurement_history(round(foam_height))
# 
#         # Visualization (if needed)
#         self.draw_irregular_edges(self.image.copy(), upper_edge, lower_edge)
    
        
        
if __name__ == "__main__":
    root = tk.Tk()
    app = FoamMeasurementApp(root, "Foam Thickness Measurement")
    try:
        root.mainloop()
    finally:
        app.cleanup()


