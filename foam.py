import cv2
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
from picamera2 import Picamera2
import paho.mqtt.client as mqtt
import time
import json

class FoamMeasurementApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
        self.picam2.start()
        time.sleep(1)  # Give the camera time to warm up
        
        self.image_lock = threading.Lock()
        self.current_image = None
        self.running = True
        self.processing_thread = threading.Thread(target=self.image_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # MQTT setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.connect("localhost", 1884, 60)
        self.mqtt_client.loop_start()
        
        self.image = None
        self.processed_image = None
        self.scale = 1.0  # pixels per mm
        self.show_edges = tk.BooleanVar(value=False)
        self.roi = [0, 0, 640, 480]  # [x, y, w, h]
        self.drawing_roi = False
        self.upper_edge = None
        self.lower_edge = None
        self.tuning_mode = False
        self.calibrating = False
        self.calibration_start = None
        self.calibration_end = None
        
        self.show_edges = tk.BooleanVar(value=True)
        
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
    
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        result_meanings = {
            0: "Connection successful",
            1: "Connection refused - incorrect protocol version",
            2: "Connection refused - invalid client identifier",
            3: "Connection refused - server unavailable",
            4: "Connection refused - bad username or password",
            5: "Connection refused - not authorized"
        }
        
        if rc in result_meanings:
            result_message = result_meanings[rc]
        else:
            result_message = f"Unknown result code: {rc}"
        
        print(f"MQTT Connection result: {result_message}")
        
        if rc == 0:
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
        self.scale_label = ttk.Label(scale_frame, text="Scale (px/mm):")
        self.scale_label.pack(side=tk.LEFT)
        self.scale_entry = ttk.Entry(scale_frame, width=10)
        self.scale_entry.insert(0, "1.0")
        self.scale_entry.pack(side=tk.LEFT)

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
        # Calibration section in tuning frame
        self.calibration_label = ttk.Label(self.tuning_frame, text="Scale Calibration:")
        self.calibration_label.pack(pady=(10,0))
        
        self.calibrate_button = ttk.Button(self.tuning_frame, text="Start Calibration", command=self.start_calibration)
        self.calibrate_button.pack(pady=5)
        
        self.calibration_info_label = ttk.Label(self.tuning_frame, text="")
        self.calibration_info_label.pack(pady=5)
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
        self.calibration_info_label.config(text="Draw a line on the image\ncorresponding to a known distance.")
        self.canvas.bind("<ButtonPress-1>", self.calibration_start_point)
        self.canvas.bind("<ButtonRelease-1>", self.calibration_end_point)

    def calibration_start_point(self, event):
        self.calibration_start = (event.x, event.y)

    def calibration_end_point(self, event):
        self.calibration_end = (event.x, event.y)
        self.calculate_scale()

    def calculate_scale(self):
        if self.calibration_start and self.calibration_end:
            pixel_distance = ((self.calibration_end[0] - self.calibration_start[0])**2 + 
                              (self.calibration_end[1] - self.calibration_start[1])**2)**0.5
            
            known_distance = simpledialog.askfloat("Input", "Enter the known distance in millimeters:", 
                                                   parent=self.window)
            
            if known_distance:
                self.scale = pixel_distance / known_distance
                self.scale_entry.delete(0, tk.END)
                self.scale_entry.insert(0, f"{self.scale:.4f}")
                self.calibration_info_label.config(text=f"Scale set to {self.scale:.4f} pixels/mm")
            else:
                self.calibration_info_label.config(text="Calibration cancelled.")
            
            self.calibrating = False
            self.calibrate_button.config(state="normal")
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<ButtonRelease-1>")
            
    def start_roi(self, event):
        if not self.tuning_mode:
            self.roi = [event.x, event.y, 0, 0]
            self.drawing_roi = True

    def draw_roi(self, event):
        if not self.tuning_mode and self.drawing_roi:
            self.roi[2] = event.x - self.roi[0]
            self.roi[3] = event.y - self.roi[1]

    def end_roi(self, event):
        if not self.tuning_mode:
            self.drawing_roi = False
            self.roi[2] = max(event.x - self.roi[0], 1)
            self.roi[3] = max(event.y - self.roi[1], 1)
            self.roi_label.config(text=f"ROI: {self.roi}")
       
    def toggle_tuning_mode(self):
        self.tuning_mode = not self.tuning_mode
        if self.tuning_mode:
            self.tuning_button.config(text="Exit Tuning Mode")
            self.tuning_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
            self.process_image()
        else:
            self.tuning_button.config(text="Enter Tuning Mode")
            self.tuning_frame.grid_forget()

    def apply_tuning(self):
        self.process_image()

    def update_video(self):
        try:
            with self.image_lock:
                if self.current_image is not None:
                    self.image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

            if self.image is not None:
                # Debugging line
                #print(f"Image shape: {self.image.shape}")  
                display_image = self.image.copy()

                if self.tuning_mode:
                    self.process_image()
                    if self.full_processed_image is not None:
                        display_image = self.full_processed_image.copy()
                elif self.show_edges.get():
                    self.process_image()
                    self.draw_edges(display_image)

                if self.calibrating:
                    if self.calibration_start and self.calibration_end:
                        cv2.line(display_image, self.calibration_start, self.calibration_end, (255, 255, 0), 2)
                    elif self.calibration_start:
                        cv2.circle(display_image, self.calibration_start, 3, (255, 255, 0), -1)

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                # Draw the ROI rectangle
                self.canvas.create_rectangle(self.roi[0], self.roi[1], 
                                             self.roi[0]+self.roi[2], 
                                             self.roi[1]+self.roi[3], 
                                             outline="red")
            else:
                print("Error: No image captured")
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

        # Find upper edge
        for i in range(h):
            if np.sum(edge_image[i]) > 0.1 * w * 255:  # Assume edge starts when 10% of row is edge
                upper_edge = i
                break
        else:
            return None, None

        # Find lower edge
        for i in range(h - 1, upper_edge, -1):
            if np.sum(edge_image[i]) > 0.1 * w * 255:  # Assume edge ends when 10% of row is edge
                lower_edge = i
                break
        else:
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

            self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            self.full_processed_image = self.image.copy()
            self.full_processed_image[y:y+h, x:x+w] = self.processed_image

            self.upper_edge, self.lower_edge = self.find_foam_edges(edges)

        except Exception as e:
            print(f"Error processing image: {e}")
            self.processed_image = None
            self.full_processed_image = None
            self.upper_edge, self.lower_edge = None, None


                
    def draw_edges(self, image):
        if image is None or self.upper_edge is None or self.lower_edge is None:
            return

        try:
            x, y, w, h = self.roi

            # Draw upper edge
            cv2.line(image, (x, y + self.upper_edge), (x + w, y + self.upper_edge), (0, 255, 0), 2)
            
            # Draw lower edge
            cv2.line(image, (x, y + self.lower_edge), (x + w, y + self.lower_edge), (0, 255, 0), 2)
            
            # Draw vertical line
            mid_x = x + w // 2
            cv2.line(image, (mid_x, y + self.upper_edge), (mid_x, y + self.lower_edge), (255, 0, 0), 2)

            # Draw text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, "Upper", (x, y + self.upper_edge - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "Lower", (x, y + self.lower_edge + 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error drawing edges: {e}")
        
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
            self.history_table.insert('', 'end', values=(timestamp, f"{thickness:.2f}"))
            
    def update_measurement_history(self, foam_height):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.measurement_history.append((timestamp, foam_height))
            if len(self.measurement_history) > 3:
                self.measurement_history.pop(0)
            self.update_history_table()
    
    def measure_foam(self):
        self.process_image()  # Process the image before measuring
        
        if self.processed_image is None or self.upper_edge is None or self.lower_edge is None:
            self.result_label.config(text="No foam detected")
            return

        try:
            self.scale = float(self.scale_entry.get())
        except ValueError:
            self.result_label.config(text="Invalid scale value")
            return

        thickness_px = self.lower_edge - self.upper_edge
        foam_height = thickness_px / self.scale
        result_text = f"Foam thickness: {foam_height:.2f} ml"
        self.result_label.config(text=result_text)

        # Update measurement history
        self.update_measurement_history(foam_height)
    
        
        
if __name__ == "__main__":
    root = tk.Tk()
    app = FoamMeasurementApp(root, "Foam Thickness Measurement")
    try:
        root.mainloop()
    finally:
        app.cleanup()
