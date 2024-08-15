import sys
import argparse
import logging
import pathlib
import json
import time
import datetime
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os

# Utils


def fix_image_size(image: np.array, expected_pixels: float = 2E6):
    ratio = np.sqrt(expected_pixels / (image.shape[0] * image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image: np.array, threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return blur_map, score, bool(score < threshold)


def pretty_blur_map(blur_map: np.array, sigma: int = 5, min_abs: float = 0.5):
    abs_image = np.abs(blur_map).astype(np.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = np.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)

def stamp_image(image, size = (780,540), display = False, save = False, save_path = 0, incr = 2.5):
    image = cv2.resize(image, size,interpolation=cv2.INTER_LINEAR)
    window_name = 'Added time stamp'
    time_date = datetime.datetime.now()
    time_date = time_date.strftime("%x") + ' ' + time_date.strftime("%X")

    # Using cv2.putText() method
    image = cv2.putText(image, time_date, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65*incr,
                        (0, 0, 0), 2*int(1*incr), cv2.LINE_AA, False)
    image = cv2.putText(image, time_date, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65*incr,
                        (0, 165, 255), int(1*incr), cv2.LINE_AA, False)

    # Displaying the image
    if display:
        print(f'Image displayed with timestamp: {time_date}')
        cv2.imshow(window_name, image)
        cv2.waitKey(0)

    # Saving the image
    if save:
        cv2.imwrite(save_path, image)
        print(f'Image saved at {save_path}')

    return image


# END





class ImageBlurApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Blur Detection Tool")
        self.root.minsize(1000, 600)

        self.threshold = 100  # Initial threshold value
        self.relative_path = ''
        self.path = ''
        self.blur_value = 0  # Initialize blur value
        self.image = []
        self.time_stamp = False

        # Frame for the base image
        self.base_frame = tk.Frame(self.root)
        self.base_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.base_image_label = tk.Label(self.base_frame, bg="gray")
        self.base_image_label.pack(padx=10, pady=10)

        # Frame for the work image
        self.work_frame = tk.Frame(self.root)
        self.work_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.work_image_label = tk.Label(self.work_frame, bg="gray")
        self.work_image_label.pack(padx=10, pady=10)

        # Frame for the button and display zone
        self.button_display_frame = tk.Frame(self.root)
        self.button_display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.browse_button = tk.Button(self.button_display_frame, text="Browse", command=self.browse_image)
        self.browse_button.pack(padx=10, pady=10)

        self.image_path_label = tk.Label(self.button_display_frame, text="Image Path: ")
        self.image_path_label.pack(padx=10, pady=10, anchor=tk.W)

        # Button to detect blur
        self.detect_blur_button = tk.Button(self.button_display_frame, text="Detect Blur", command=self.detect_blur)
        self.detect_blur_button.pack(padx=10, pady=10)

        # Input zone for threshold
        self.threshold_label = tk.Label(self.button_display_frame, text="Threshold:")
        self.threshold_label.pack(padx=10, pady=5, anchor=tk.W)

        self.threshold_entry = tk.Entry(self.button_display_frame)
        self.threshold_entry.insert(tk.END, str(self.threshold))
        self.threshold_entry.pack(padx=10, pady=5)

        self.update_threshold_button = tk.Button(self.button_display_frame, text="Update Threshold",
                                                 command=self.update_threshold)
        self.update_threshold_button.pack(padx=10, pady=5)

        # Frame to hold blur value display and timestamp button
        self.blur_value_frame = tk.Frame(self.button_display_frame)
        self.blur_value_frame.pack(padx=10, pady=5, anchor=tk.W)

        # Label to display "Blur Computed:"
        self.blur_value_label = tk.Label(self.blur_value_frame, text="Blur Computed:")
        self.blur_value_label.pack(side=tk.LEFT)

        # Label to display the value of self.blur_value
        self.blur_value_display = tk.Label(self.blur_value_frame, text=str(self.blur_value))
        self.blur_value_display.pack(side=tk.LEFT)

        # Button to add timestamp
        self.add_timestamp_button = tk.Button(self.button_display_frame, text="Add Timestamp", command=self.add_time_stamp)
        self.add_timestamp_button.pack(side=tk.LEFT, padx=10, pady=5)

        # Label to display the state of the time_stamp attribute
        self.time_stamp_label = tk.Label(self.button_display_frame, text="Time Stamp: ")
        self.time_stamp_label.pack(side=tk.LEFT, padx=10, pady=5)

        self.time_stamp_display = tk.Label(self.button_display_frame, text="Deactivated")
        self.time_stamp_display.pack(side=tk.LEFT, padx=5, pady=5)

    def browse_image(self):
        file_path = filedialog.askopenfilename()
        self.path = file_path
        if file_path:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_LINEAR)
            self.image = image
            # Resize the frame to fit the image
            self.base_frame.configure(width=image.shape[1], height=image.shape[0])
            image_PIL = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image_PIL)

            self.base_image_label.configure(image=photo)
            self.base_image_label.image = photo

            # Get the relative path
            self.relative_path = os.path.relpath(file_path)
            self.image_path_label.config(text="Image Path: " + self.relative_path)

            self.root.update_idletasks()  # Force update to make sure the frame size is updated

            # Store the image data for detecting blur
            self.work_image_data = image

    def detect_blur(self):
        if hasattr(self, 'work_image_data'):
            image = cv2.imread(self.path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_LINEAR)

            if self.time_stamp:
                image = stamp_image(image, (400,400), incr = 1)

            # Begin Blur Detection
            blur_map, score, blurry = estimate_blur(image, threshold=self.threshold)

            logging.info(f'image_path: {self.relative_path} score: {score} blurry: {blurry}')

            results = ({'input_path': str(self.relative_path), 'score': score, 'blurry': blurry})
            if blurry:
                text = 'Blur Detected'
            else:
                text = 'Image Clear'
            # Add Information on the Image
            image = cv2.putText(image, "{}: {:.2f}".format(text, score), (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Resize the frame to fit the image
            self.work_frame.configure(width=image.shape[1], height=image.shape[0])
            image_PIL = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image_PIL)

            self.work_image_label.configure(image=photo)
            self.work_image_label.image = photo

            self.root.update_idletasks()  # Force update to make sure the frame size is updated

            # Store the image data for detecting blur
            self.work_image_data = image
            self.blur_value = score
            self.update_blur_value_display()  # Update blur value display

    def update_threshold(self):
        try:
            self.threshold = float(self.threshold_entry.get())
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid threshold value. Please enter a number.")
            return

    def update_blur_value_display(self):
        self.blur_value_display.config(text=str(self.blur_value))  # Update blur value display

    def add_time_stamp(self):
        # Add timestamp functionality
        if self.time_stamp:
            self.time_stamp = False
            self.time_stamp_display.config(text="Deactivated")
        else:
            self.time_stamp = True
            self.time_stamp_display.config(text="Activated")


def main():
    root = tk.Tk()
    app = ImageBlurApp(root)

    # Center each section in 1/3 of the window
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)

    root.mainloop()


if __name__ == "__main__":
    main()