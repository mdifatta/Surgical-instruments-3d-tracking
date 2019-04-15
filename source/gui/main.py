from tkinter import *
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy.spatial import distance
import os
from matplotlib import pyplot as plt
import pandas as pd


class MainWindow:

    def __init__(self, main, img_width, img_height, raw_frames, photo_images, frames_name, output_name):

        self.root = main

        # list of points
        self.points = []

        # window's frame
        self.main_frame = Frame(main, height=img_height, width=img_width + 90, bd=15, relief=FLAT)

        # buttons' space
        self.cmd_frame = Frame(self.main_frame, height=img_height, width=90, bd=3, relief=SUNKEN)

        # canvas for image
        self.canvas = Canvas(self.main_frame, width=img_width, height=img_height, highlightthickness=0, bd=0)
        self.canvas.bind("<Button 1>", self.get_coord)

        # images
        self.my_images = photo_images

        self.my_frames = raw_frames
        self.current_frame = 0

        # frames' file names
        self.filenames = frames_name

        # set first image on canvas
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.my_images[self.current_frame])

        # name of the current frame
        self.curr_frame_name = Label(self.cmd_frame, text=self.filenames[self.current_frame])

        # frame count
        self.count = Label(self.cmd_frame, text='%d/%d' % (self.current_frame, len(self.filenames)))

        # button to change image
        self.button = Button(self.cmd_frame, text='Next>>', command=self.change_frame)

        # button to invalid frame
        self.invalid_btn = Button(self.cmd_frame, text='Invalid', command=self.invalid_frame)

        # button to clear marked points
        self.clear_btn = Button(self.cmd_frame, text='Clear', command=self.clear_points)

        self.main_frame.pack(fill=BOTH)
        self.canvas.pack(side=LEFT)
        self.cmd_frame.pack(side=RIGHT, fill=BOTH)
        self.button.pack(expand=True)
        self.invalid_btn.pack(side='bottom')
        self.clear_btn.pack(side='bottom')
        self.curr_frame_name.pack(expand=True)
        self.count.pack(expand=True)

        self.output_file = '../../data/targets/' + output_name + '.csv'

    def change_frame(self):
        if len(self.points) == 2:
            # save current coordinates
            f = open(self.output_file, 'a')
            dist = distance.euclidean(self.points[0], self.points[1])
            txt = '%s;%d;%s;%s;%.1f\n' % (
                self.filenames[self.current_frame], 1, self.points[0], self.points[1], dist)
            f.write(txt)
            f.close()

            # next image
            self.current_frame += 1

            # return to first image
            if self.current_frame == len(self.my_images):
                messagebox.showinfo('This is the End', 'The current sequence of frames is ended.')
                self.root.destroy()
                return

            # change image
            self.canvas.itemconfig(self.image_on_canvas, image=self.my_images[self.current_frame])
            self.curr_frame_name.configure(text=self.filenames[self.current_frame])
            self.count.configure(text = '%d/%d' % (self.current_frame, len(self.filenames)))
            self.points.clear()

    def get_coord(self, event):
        if len(self.points) < 2:
            x = event.x * 2
            y = event.y * 2
            # outputting x and y coord to console
            print(x, y)
            self.points.append((x, y))

        if len(self.points) == 2:
            # ########################## showing points for testing ###############################
            blank = np.copy(self.my_frames[self.current_frame])
            blank = cv2.resize(blank, dsize=(0, 0), fx=2, fy=2)
            res = cv2.circle(blank, self.points[0], 6, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            res = cv2.circle(res, self.points[1], 6, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            plt.figure()
            plt.imshow(res)
            plt.show()
            # #####################################################################################
            print(self.points)

    def invalid_frame(self):
        # save current coordinates
        f = open(self.output_file, 'a')
        txt = '%s;%d;%d;%d;%d\n' % (self.filenames[self.current_frame], 0, -1, -1, -1)
        f.write(txt)
        f.close()

        # next image
        self.current_frame += 1

        # return to first image
        if self.current_frame == len(self.my_images):
            messagebox.showinfo('This is the End', 'The current sequence of frames is ended.')
            self.root.destroy()
            return

        # change image
        self.canvas.itemconfig(self.image_on_canvas, image=self.my_images[self.current_frame])
        self.curr_frame_name.configure(text=self.filenames[self.current_frame])
        self.count.configure(text='%d/%d' % (self.current_frame, len(self.filenames)))
        self.points.clear()

    def clear_points(self):
        self.points.clear()


points = []


def simple_gui():
    frames = os.listdir('./data/')

    window = Tk()

    raw = cv2.cvtColor(cv2.imread(frames[index]), cv2.COLOR_BGR2RGB)
    img = cv2.resize(raw, (0, 0), fx=.5, fy=.5)

    image_height = img.shape[0]
    image_width = img.shape[1]
    cmd_height = image_height
    cmd_width = 90

    frame = Frame(window, height=image_height, width=image_width + cmd_width, bd=15, relief=FLAT)

    cmd_frame = Frame(frame, height=cmd_height, width=cmd_width, bd=1, relief=FLAT)

    canvas = Canvas(frame, width=image_width, height=image_height, highlightthickness=0, bd=0)

    frame.pack(fill=BOTH)
    cmd_frame.pack(side=RIGHT)
    canvas.pack(side=LEFT)

    next_btn = Button(cmd_frame, text='Next>>')
    next_btn.pack(side='bottom')

    # adding the image
    image = ImageTk.PhotoImage(image=Image.fromarray(img))
    canvas.create_image(0, 0, image=image, anchor="nw")

    # function to be called when mouse is clicked
    def get_coords(event):

        if len(points) < 2:
            x = event.x * 2
            y = event.y * 2
            # outputting x and y coords to console
            print(x, y)
            points.append((x, y))

        if len(points) == 2:
            blank = np.copy(raw)
            res = cv2.circle(blank, points[0], 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            res = cv2.circle(res, points[1], 1, (0, 0, 255), -1)
            res = cv2.line(res, points[0], points[1], (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.imshow('prova-1', res)

    def change_frame():
        global index
        index = index + 1

    # mouse click event
    canvas.bind("<Button 1>", get_coords)

    next_btn.bind('<Button-1>', change_frame)

    window.mainloop()


if __name__ == '__main__':
    # simple_gui()

    root = Tk()

    folders_names = os.listdir('../../data/datasets/distance_frames_folders/')

    answer = simpledialog.askstring("Frames folder",
                                    "Which frames' folder do you want to use?\n"+'\n'.join(folders_names),
                                    parent=root)

    path = '../../data/datasets/distance_frames_folders/' + answer + '/'

    if answer is not None and answer != '':
        file_names = os.listdir(path)

        exists = os.path.isfile('../../data/targets/' + answer + '.csv')

        if exists:
            df = pd.read_csv('../../data/targets/' + answer + '.csv', sep=';')
            df_names = df['file']
            csv_names_set = set(df_names)
            file_names_set = set(file_names)
            file_names = list(file_names_set.difference(csv_names_set))
        else:
            f = open('../../data/targets/' + answer + '.csv', 'a+')
            f.write('file;valid;p1;p2;dist\n')
            f.close()

        raw_images = []
        photo_imgs = []
        for fr in file_names:
            tmp = cv2.resize(cv2.cvtColor(cv2.imread(path + fr), cv2.COLOR_BGR2RGB), (0, 0), fx=.5, fy=.5)
            raw_images.append(tmp)
            photo_imgs.append(ImageTk.PhotoImage(image=Image.fromarray(tmp)))

        MainWindow(root, raw_images[0].shape[1], raw_images[0].shape[0], raw_images, photo_imgs, file_names, answer)
        root.mainloop()
    else:
        print('You need to choose a folder.')

# TODO: load each single image when needed
