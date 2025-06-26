import customtkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
import tracemalloc
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pathlib import Path
from app import load, write_results, clear_results_file ,process_image, compare_masks, decode_bitmap,get_bitmapdata_supervisely ,create_blank_map, save_results, convert_box_to_bitmapdata, combine_box_bitmapdata, load_boxes_from_path  # Import the load function from app.py
import time


class PictureFrame(tk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.CTkCanvas(self, width=600, height=600)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.box_drawn = False
        self.setting_frame = None
        self.image = None
        self.bbox = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rect_id = None
        self.original_image = None
        self.mask_image = None
        self.current_image = None
        self.detection_boxes = None
        self.toggle_r = True
        self.toggle_m = True
        
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
    def set_setting_frame(self, setting_frame):
        self.setting_frame = setting_frame
    def toggle_mask(self):
        if self.toggle_m == True:
            self.toggle_m = False
            self.current_image = self.original_image
        else:
            self.toggle_m = True
            self.current_image = self.mask_image
        self.update_picture(self.current_image)
        if self.toggle_r == True:
            self.draw_rectangles()
    def update_picture(self, image):
        self.canvas.delete("all")
        self.box_drawn = False
        self.current_image = image
        image = image.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.image)
    def toggle_rectangle(self):
        if self.toggle_r == True:
            self.toggle_r = False
            self.remove_rectangles()
        else:
            self.toggle_r = True
            self.draw_rectangles()
    def remove_rectangle(self):
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.box_drawn = False
    def draw_rectangles(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = ImageTk.PhotoImage(self.current_image).width()
        image_height = ImageTk.PhotoImage(self.current_image).height()

        scale_x = canvas_width / image_width
        scale_y = canvas_height / image_height
        for box in self.detection_boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
    def remove_rectangles(self):
        self.update_picture(self.current_image)
        
        
    def on_mouse_down(self, event):
        if self.setting_frame.comboBoxDetection.get() == "Manual" and not self.box_drawn:
            self.start_x = event.x
            self.start_y = event.y
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_mouse_drag(self, event):
        if self.setting_frame.comboBoxDetection.get() == "Manual" and not self.box_drawn:
            self.end_x = event.x
            self.end_y = event.y
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, self.end_x, self.end_y)

    def on_mouse_up(self, event):
        if self.setting_frame.comboBoxDetection.get() == "Manual" and not self.box_drawn:
            self.end_x = event.x
            self.end_y = event.y
            print(f"Bounding box: ({self.start_x}, {self.start_y}) to ({self.end_x}, {self.end_y})")
            self.box_drawn = True
            self.pass_coordinates_to_method()

    def pass_coordinates_to_method(self):
        if self.setting_frame.comboBoxDetection.get() == "Manual":
            image_width = ImageTk.PhotoImage(self.current_image).width()
            image_height = ImageTk.PhotoImage(self.current_image).height()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            scale_x = image_width / canvas_width
            scale_y = image_height / canvas_height

            x1 = int(self.start_x * scale_x)
            y1 = int(self.start_y * scale_y)
            x2 = int(self.end_x * scale_x)
            y2 = int(self.end_y * scale_y)

            print(f"Image coordinates: ({x1}, {y1}) to ({x2}, {y2})")
            self.bbox = np.array([x1, y1, x2, y2]) 
    def get_bbox(self):
        if self.setting_frame.comboBoxDetection.get() == "Manual":
            print(f"get_bbox: {self.bbox}")
            return self.bbox
class SettingFrame(tk.CTkFrame):
    def __init__(self, master, picture_frame):
        super().__init__(master)
        self.running = True
        self.pictureFrame = picture_frame
        
        '''
        Ideal structure:
        0. Detection model Label
        1. Detection model ComboBox
        2. Segmentation model Label
        3. Segmentation model ComboBox
        4. Select Dataset type label
        5. Select Dataset type ComboBox
        6. Sensitivity label
        7. Sensitivity Slider
        8. Select Dataset button
        8.5 Or label
        9. Select Image button
        10. Select Annotations button
        11. Start button - red
        12. Toggle Bboxes button - green
        13. Toggle Masks button - green
        '''
        
        self.labelDetection = tk.CTkLabel(self, text="Detection model:")
        self.labelDetection.grid(row = 0, column = 0, padx = 10, pady = (10, 0), sticky = "w")
        
        self.comboBoxDetection = tk.CTkOptionMenu(self, values=["----","YOLOv8", "OpenCV", "MaskRCNN", "Mask2Former", "Head", "Manual", "Retina"], command=self.on_detection_change)
        self.comboBoxDetection.grid(row = 1, column = 0, padx = 10, pady = (0, 10), sticky = "w")
        
    
            
        self.labelSegmentation = tk.CTkLabel(self, text="Segmentation model:")
        self.labelSegmentation.grid(row = 2, column = 0, padx = 10, pady = (10, 0), sticky = "w")
        
        self.comboBoxSegmentation = tk.CTkOptionMenu(self, values=["----","SAM", "OpenCV", "MaskRCNN", "Mask2Former"], command=self.on_segmentation_change)
        self.comboBoxSegmentation.grid(row = 3, column = 0, padx = 10, pady = (0, 10), sticky = "w")
        
        self.labelAnnotationType = tk.CTkLabel(self, text="Select Annotation Type:")
        self.labelAnnotationType.grid(row = 4, column = 0, padx = 10, pady = (10, 0), sticky = "w")
        self.comboBoxAnnotationType = tk.CTkOptionMenu(self, values=["segmentation", "detection"])
        self.comboBoxAnnotationType.grid(row = 5, column = 0, padx = 10, pady = (0, 10), sticky = "w")
        
        self.sliderValue = tk.StringVar(value="Sensitivity: 50")
        self.labelSensitivity = tk.CTkLabel(self, textvariable=self.sliderValue)
        self.labelSensitivity.grid(row = 6, column = 0, padx = 10, pady = (10, 0), sticky = "w")
        
        self.sliderSensitivity = tk.CTkSlider(self, from_=0, to=100, command=self.update_label)
        self.sliderSensitivity.grid(row = 7, column = 0, padx = 10, pady = (0, 10), sticky = "w")
        self.sliderSensitivity.set(50)
        
        
        self.buttonAdd = tk.CTkButton(self, text="Select Dataset", width=30, height=30, command=self.add_button)
        self.buttonAdd.grid(row = 8, column = 0, padx = 10, sticky = "w")
        
        self.labelOr = tk.CTkLabel(self, text="Or")
        self.labelOr.grid(row = 9, column = 0, padx = 20, sticky = "w")
        
        self.buttonAddImage = tk.CTkButton(self, text="Select Image", width=30, height=30, command=self.add_image_button)
        self.buttonAddImage.grid(row = 10, column = 0, padx = 10, sticky = "w")
        
        self.buttonRemove = tk.CTkButton(self, text="Select Annotation", width=30, height=30, command=self.add_annotations_button)
        self.buttonRemove.grid(row = 11, column = 0, padx = 10, pady = (10, 10), sticky = "w")
        
        
        
        self.buttonStart = tk.CTkButton(self, text="Start", command=self.start_button, fg_color="red")
        self.buttonStart.grid(row = 12, column = 0, padx = 10, pady = (10, 10), sticky = "w")
        
        self.buttonToggleRectangle = tk.CTkButton(self, text="Toggle Bboxes", width=30, height=30, command=self.toggle_rectangle, fg_color="green")
        self.buttonToggleRectangle.grid(row = 13, column = 0, padx = 10, pady = (10, 10), sticky = "w")
        
        self.buttonToggleMask = tk.CTkButton(self, text="Toggle Masks", width=30, height=30, command=self.toggle_mask, fg_color="green")
        self.buttonToggleMask.grid(row = 14, column = 0, padx = 10, sticky = "w")
        
        
    def toggle_rectangle(self):
        if folder_path is None:
            self.pictureFrame.toggle_rectangle()
    
    def toggle_mask(self):
        if folder_path is None:
            self.pictureFrame.toggle_mask()
        
    def on_detection_change(self, value):
        if value == "MaskRCNN":
            self.comboBoxSegmentation.set("MaskRCNN")
        elif value == "Mask2Former":
            self.comboBoxSegmentation.set("Mask2Former")
        elif self.comboBoxSegmentation.get() == "MaskRCNN" or self.comboBoxSegmentation.get() == "Mask2Former":
            self.comboBoxSegmentation.set("----")
        self.pictureFrame.remove_rectangle()
    def on_segmentation_change(self, value):
        if value == "MaskRCNN":
            self.comboBoxDetection.set("MaskRCNN")
        elif value == "Mask2Former":
            self.comboBoxDetection.set("Mask2Former")
        
    def update_label(self, value):
        self.sliderValue.set(f"Sensitivity: {int(value)}")
        
    def add_button(self):
        global folder_path, file_path, annotations_folder_path
        annotations_folder_path = None
        folder_path = filedialog.askdirectory(
            title="Select a Folder"
        )
        if folder_path:
            print(f"Selected folder: {folder_path}")
            file_path = None
    def add_annotations_button(self):
        global annotations_folder_path
        annotations_folder_path = filedialog.askdirectory(
            title="Select an Annotations Folder"
        )
        if annotations_folder_path:
            print(f"Selected annotations folder: {annotations_folder_path}")
    def add_image_button(self):
        global file_path, folder_path, annotations_folder_path
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            print(f"Selected image: {file_path}")
            folder_path = None
            annotations_folder_path = None
            self.pictureFrame.update_picture(Image.open(file_path))
    def start_button(self):
        if not folder_path and not file_path:
            print("No folder or image selected.")
            return
        if folder_path:
            self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if file_path:
            self.image_files = [file_path]
        print("selected values" , self.comboBoxDetection.get(), self.comboBoxSegmentation.get(), self.sliderSensitivity.get())
        print("file count", self.image_files)
        self.process_next_image(0)   
    
    def process_next_image(self, index):
        if(index == 0):
            clear_results_file(self.comboBoxDetection.get(), self.comboBoxSegmentation.get())
        if index < len(self.image_files) and self.running:
            filename = self.image_files[index]
            if not file_path:    
                image_path = os.path.join(folder_path, filename)
            else:
                image_path = file_path
            image = Image.open(image_path)
            self.pictureFrame.update_picture(image)
            self.pictureFrame.original_image = image
            mask_path = "none"
            if annotations_folder_path is not None:
                mask_path = os.path.join(annotations_folder_path, f"{filename}.json")
            
            self.after(2000, self.process_image, image_path, mask_path, index)

    def process_image(self, image_path, mask_path, index):
        if os.path.exists(image_path):
            tracemalloc.start()
            loadingTime = time.time()
            if self.comboBoxDetection.get() != "Manual":
                processed_image, processed_mask, detection_boxes = process_image(image_path, self.sliderSensitivity.get()/100, self.comboBoxDetection.get(), self.comboBoxSegmentation.get())
            else:
                processed_image, processed_mask, detection_boxes = process_image(image_path, self.sliderSensitivity.get()/100, self.comboBoxDetection.get(), self.comboBoxSegmentation.get(), self.pictureFrame.get_bbox())
            endTime = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            loadingTime = endTime - loadingTime
            print(f"Loading time for models {self.comboBoxDetection.get()}, {self.comboBoxSegmentation.get()}: {loadingTime}")
            dice = 0
            iou = 0
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            if os.path.exists(mask_path):
                create_blank_map()
                processed_box = 0
                original_box = 0
                maskbmp = 0
                if self.comboBoxAnnotationType.get() == "segmentation":
                    maskbmp = get_bitmapdata_supervisely(mask_path)
                    dice, iou, tp, fp, tn, fn = compare_masks(processed_mask, maskbmp, processed_box, original_box, True, self.comboBoxAnnotationType.get())
                    
                elif self.comboBoxAnnotationType.get() == "detection":
                    original_boxes = load_boxes_from_path(mask_path)
                    if detection_boxes:
                        pred_boxes_tensor = torch.tensor(detection_boxes, dtype=torch.float)
                        pred_labels = torch.ones(len(detection_boxes), dtype=torch.int)
                        scores = torch.ones(len(detection_boxes), dtype=torch.float)
                    else:
                        pred_boxes_tensor = torch.zeros((0, 4), dtype=torch.float)
                        pred_labels = torch.tensor([], dtype=torch.int)
                        scores = torch.tensor([], dtype=torch.float)
                    gt_boxes_tensor = torch.tensor(original_boxes, dtype=torch.float)
                    gt_labels = torch.ones(len(original_boxes), dtype=torch.int)

                    preds = [{
                        "boxes": pred_boxes_tensor,
                        "scores": scores,
                        "labels": pred_labels
                    }]
                    targets = [{
                        "boxes": gt_boxes_tensor,
                        "labels": gt_labels
                    }]

                    metric = MeanAveragePrecision()
                    metric.update(preds, targets)
                    results = metric.compute()
                    print(f"mAP50: {results['map_50']}")
                    iou = results['map_50']
                
                
            self.pictureFrame.mask_image = processed_image
            self.pictureFrame.update_picture(self.pictureFrame.mask_image)
            self.pictureFrame.detection_boxes = detection_boxes
            self.pictureFrame.draw_rectangles()
            write_results(image_path, self.comboBoxDetection.get(), self.comboBoxSegmentation.get(), loadingTime, dice, iou, tp, fp, tn, fn, peak, processed_image.size[1], processed_image.size[0])
            save_results(image_path, processed_mask, dice, iou, self.comboBoxSegmentation.get(), self.comboBoxDetection.get())   
        self.after(2000, self.process_next_image, index + 1) 
                
class GUI(tk.CTk):
    def __init__(self):
        super().__init__()
        tk.set_appearance_mode("System")
        tk.set_default_color_theme("dark-blue")
        self.title("BC Praca 1.0")
        self.geometry("900x650")
        self.maxsize(900, 650)
        self.minsize(900, 650)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=6)
        self.grid_rowconfigure(0, weight=1)
        
        self.pictureFrame = PictureFrame(self)
        self.pictureFrame.grid(row=0, column=1, padx=10, pady=10, sticky="nswe")
        
        self.settingFrame = SettingFrame(self, self.pictureFrame)
        self.settingFrame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        
        self.pictureFrame.set_setting_frame(self.settingFrame)
        self.bind("<Escape>", self.stop_process)
        load()
    def stop_process(self, event=None):
        self.settingFrame.running = False
        print("Process stopped by key press.")

app = GUI()
app.mainloop()